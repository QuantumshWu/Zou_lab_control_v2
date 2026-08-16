"""Repeat-mean site-brightness feedback; no hidden-plant inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from zlc_data import SCAN_POINT, SPATIAL_X, SPATIAL_Y, AxisId, PointColumn
from zlc_durable import unique_path
from zlc_pulse import PulseSequence
from zlc_runtime import DatasetCoverage, DatasetOutputDeclaration, LiveDatasetOutput

from zlc_atom.data import snapshot_from_array
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
from zlc_atom.nodes.scan import ScanLiveSlot
from zlc_atom.nodes.scan.source import wait_for_board


SLM_PHASE_ARTIFACT_CONTRACT = "zlc.slm.phase.v1"
READOUT_AVERAGE_OUTPUT = DatasetOutputDeclaration(
    "readout_average", "slm-feedback.readout-average.v1"
)
CANDIDATE_PHASE_OUTPUT = DatasetOutputDeclaration(
    "candidate_phase", "slm-feedback.candidate-phase.v1"
)
UNIFORMITY_HISTORY_OUTPUT = DatasetOutputDeclaration(
    "uniformity_history", "slm-feedback.uniformity-history.v1"
)
_TARGET_RATIO = 1.01
_VALIDATION_RELATIVE_SEM = 0.005
_SOLVE_ITERATIONS = 8
_FEEDBACK_EXPONENT = 0.45
_READOUT_FRAME = 1
_SHOT_CHUNK = 128
_GEOMETRY_TOLERANCE_FRACTION = 0.25
_MAX_NORMALIZED_AFFINE_DEVIATION = 0.25
_MAX_NORMALIZED_AFFINE_CONDITION = 3.0
_MAX_CROSS_AXIS_SPAN_FRACTION = 0.25


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


def _support(
    target: np.ndarray, calibration: TrapCalibration
) -> tuple[np.ndarray, np.ndarray]:
    """Return target support coordinates aligned to Calibration site order."""

    rows, columns = np.nonzero(target > 0.0)
    if len(rows) != calibration.n_sites:
        raise ValueError(
            "SLM target support count differs from the Calibration SiteMap count"
        )
    if not len(rows):
        raise ValueError("SLM feedback target support is empty")
    if len(rows) == 1:
        return rows, columns

    centers = np.asarray(calibration.site_map.centers_xy, dtype=float)
    separations = np.sqrt(
        np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2, axis=2)
    )
    separations[np.diag_indices_from(separations)] = np.inf
    minimum_spacing = float(np.min(separations))
    if not np.isfinite(minimum_spacing) or minimum_spacing <= 0.0:
        raise ValueError("Calibration SiteMap geometry is ambiguous")

    target_xy = np.column_stack((columns, rows)).astype(float, copy=False)
    normalized_target = np.zeros_like(target_xy)
    normalized_centers = np.zeros_like(centers)
    for axis in range(2):
        authored = target_xy[:, axis]
        measured = centers[:, axis]
        authored_span = float(np.ptp(authored))
        measured_span = float(np.ptp(measured))
        if authored_span > 0.0:
            if measured_span == 0.0:
                raise ValueError(
                    "Calibration SiteMap geometry has unmatched target support"
                )
            normalized_target[:, axis] = (
                authored - float(np.min(authored))
            ) / authored_span
            normalized_centers[:, axis] = (
                measured - float(np.min(measured))
            ) / measured_span

    initial_cost = np.sum(
        (
            normalized_target[:, np.newaxis, :]
            - normalized_centers[np.newaxis, :, :]
        )
        ** 2,
        axis=2,
    )
    target_indices, calibration_indices = linear_sum_assignment(initial_cost)
    calibration_for_target = np.empty(len(rows), dtype=np.intp)
    calibration_for_target[target_indices] = calibration_indices
    design = np.column_stack((target_xy, np.ones(len(rows), dtype=float)))
    for _iteration in range(4):
        affine, *_unused = np.linalg.lstsq(
            design, centers[calibration_for_target], rcond=None
        )
        predicted = design @ affine
        cost = np.sum(
            (predicted[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2,
            axis=2,
        )
        target_indices, calibration_indices = linear_sum_assignment(cost)
        updated_order = np.empty(len(rows), dtype=np.intp)
        updated_order[target_indices] = calibration_indices
        if np.array_equal(updated_order, calibration_for_target):
            break
        calibration_for_target = updated_order

    affine, *_unused = np.linalg.lstsq(
        design, centers[calibration_for_target], rcond=None
    )
    predicted = design @ affine
    matched_centers = centers[calibration_for_target]
    for axis in range(2):
        authored = target_xy[:, axis]
        if float(np.ptp(authored)) > 0.0 and float(
            np.sum(
                (authored - np.mean(authored))
                * (matched_centers[:, axis] - np.mean(matched_centers[:, axis]))
            )
        ) <= 0.0:
            raise ValueError("Calibration SiteMap geometry has unmatched target support")
    distances = np.sqrt(
        np.sum(
            (predicted[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2,
            axis=2,
        )
    )
    nearby = distances <= _GEOMETRY_TOLERANCE_FRACTION * minimum_spacing
    if np.any(np.sum(nearby, axis=1) > 1) or np.any(np.sum(nearby, axis=0) > 1):
        raise ValueError("Calibration SiteMap geometry is ambiguous")
    if not np.all(np.sum(nearby, axis=1) == 1) or not np.all(
        np.sum(nearby, axis=0) == 1
    ):
        raise ValueError("Calibration SiteMap geometry has unmatched target support")
    calibration_for_target = np.argmax(nearby, axis=1).astype(
        np.intp, copy=False
    )
    target_spans = np.ptp(target_xy, axis=0)
    matched_spans = np.ptp(centers[calibration_for_target], axis=0)
    if target_spans[0] > 0.0 and target_spans[1] == 0.0:
        tilted = (
            matched_spans[1] / matched_spans[0]
            > _MAX_CROSS_AXIS_SPAN_FRACTION
        )
    elif target_spans[0] == 0.0 and target_spans[1] > 0.0:
        tilted = (
            matched_spans[0] / matched_spans[1]
            > _MAX_CROSS_AXIS_SPAN_FRACTION
        )
    else:
        tilted = False
    if tilted:
        raise ValueError(
            "Calibration SiteMap differs from the trusted apparatus orientation"
        )
    if np.linalg.matrix_rank(target_xy - np.mean(target_xy, axis=0)) == 2:
        normalized_design = np.column_stack(
            (normalized_target, np.ones(len(rows), dtype=float))
        )
        normalized_affine, *_unused = np.linalg.lstsq(
            normalized_design,
            normalized_centers[calibration_for_target],
            rcond=None,
        )
        linear = normalized_affine[:2]
        determinant = float(np.linalg.det(linear))
        condition = float(np.linalg.cond(linear))
        if (
            not np.all(np.isfinite(linear))
            or determinant <= 0.0
            or not np.isfinite(condition)
            or condition > _MAX_NORMALIZED_AFFINE_CONDITION
            or float(np.max(np.abs(linear - np.eye(2))))
            > _MAX_NORMALIZED_AFFINE_DEVIATION
        ):
            raise ValueError(
                "Calibration SiteMap differs from the trusted apparatus orientation"
            )
    target_for_calibration = np.empty(len(rows), dtype=np.intp)
    target_for_calibration[calibration_for_target] = np.arange(
        len(rows), dtype=np.intp
    )
    return rows[target_for_calibration], columns[target_for_calibration]


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
            raise ValueError(
                "all calibrated sites require finite usable dark/bright BOX calibration"
            )
        self._rows, self._columns = _support(frozen_target, calibration)
        self.camera, self.sequencer, self.slm = camera, sequencer, slm
        self.camera_key, self.sequencer_key, self.slm_key = camera_key, sequencer_key, slm_key
        self.signal_plane, self.calibration, self.model = signal_plane, calibration, model
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
        self._generation = ""
        self._publication_revision = 0
        self._live_slot: ScanLiveSlot | None = None
        self._active_readout_average: np.ndarray | None = None
        self._active_readout_shots = 0

    @property
    def dataset_output_declarations(self) -> tuple[DatasetOutputDeclaration, ...]:
        return (
            READOUT_AVERAGE_OUTPUT,
            CANDIDATE_PHASE_OUTPUT,
            UNIFORMITY_HISTORY_OUTPUT,
        )

    def _candidate_metadata(
        self,
        *,
        candidate: int,
        status: str,
        history: list[dict[str, object]],
    ) -> dict[str, object]:
        return {
            "calibration_path": str(self.calibration_path),
            "target_path": str(self.target_path),
            "pulse_path": str(self.pulse_path),
            "named_devices": {
                "camera": self.camera_key,
                "sequencer": self.sequencer_key,
                "slm": self.slm_key,
            },
            "candidate": int(candidate),
            "status": str(status),
            "measurement": (
                history[-1]
                if history and history[-1]["iteration"] == candidate
                else None
            ),
            "history": history,
            "updates": len(history),
        }

    def _publish_candidate(
        self,
        *,
        phase: np.ndarray,
        readout_average: np.ndarray,
        artifact_path: Path,
        candidate: int,
        stage: str,
        shots: int,
        uniformity_ratio: float | None,
        history: list[dict[str, object]],
    ) -> None:
        slot = self._live_slot
        if slot is None:
            raise RuntimeError("SLM feedback live outputs are not attached")
        canonical = canonical_phase(phase, self.slm.shape_yx)
        image = np.asarray(readout_average, dtype=float)
        if image.shape != self.calibration.frame_contract.image_shape:
            raise ValueError("feedback readout average differs from the calibrated image shape")
        self._publication_revision += 1
        revision = self._publication_revision
        record: dict[str, object] = {
            "candidate": int(candidate),
            "stage": str(stage),
            "shots": int(shots),
            "artifact_path": str(artifact_path),
            "uniformity_ratio": (
                None if uniformity_ratio is None else float(uniformity_ratio)
            ),
            "phase_role": "canonical phase applied while readout_average was acquired",
        }
        curve_candidates = tuple(
            float(index) for index in range(1, self.max_updates + 1)
        )
        curve_ratios = np.full(self.max_updates, np.nan, dtype="<f8")
        for item in history:
            ratio = item.get("uniformity_ratio")
            if ratio is not None:
                curve_ratios[int(item["iteration"]) - 1] = float(ratio)
        coverage = DatasetCoverage(1, 1)
        slot.publish(
            {
                READOUT_AVERAGE_OUTPUT.name: LiveDatasetOutput(
                    READOUT_AVERAGE_OUTPUT,
                    snapshot_from_array(
                        image[None],
                        producer=self.instance_id,
                        signal=READOUT_AVERAGE_OUTPUT.name,
                        roles=(SPATIAL_Y, SPATIAL_X),
                        generation=self._generation,
                        revision=revision,
                    ),
                    coverage,
                    record,
                ),
                CANDIDATE_PHASE_OUTPUT.name: LiveDatasetOutput(
                    CANDIDATE_PHASE_OUTPUT,
                    snapshot_from_array(
                        canonical[None],
                        producer=self.instance_id,
                        signal=CANDIDATE_PHASE_OUTPUT.name,
                        roles=(SPATIAL_Y, SPATIAL_X),
                        value_unit="rad",
                        generation=self._generation,
                        revision=revision,
                    ),
                    coverage,
                    record,
                ),
                UNIFORMITY_HISTORY_OUTPUT.name: LiveDatasetOutput(
                    UNIFORMITY_HISTORY_OUTPUT,
                    snapshot_from_array(
                        curve_ratios[None],
                        producer=self.instance_id,
                        signal=UNIFORMITY_HISTORY_OUTPUT.name,
                        roles=(SCAN_POINT,),
                        point_columns={
                            SCAN_POINT: PointColumn(
                                AxisId("slm_feedback.candidate"),
                                "candidate",
                                SCAN_POINT,
                                PointColumn.NUMERIC,
                                curve_candidates,
                            )
                        },
                        generation=self._generation,
                        revision=revision,
                        validity=np.isfinite(curve_ratios)[None],
                    ),
                    DatasetCoverage(self.max_updates, self.max_updates),
                    record,
                ),
            }
        )

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
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        tuple[int, ...],
        tuple[int, ...],
        np.ndarray,
    ]:
        contract = self.calibration.frame_contract
        if contract.exposure_seconds is None:
            raise ValueError("calibration does not record its readout exposure")
        requested = self.shots if shots is None else int(shots)
        if requested < 2:
            raise ValueError("qCMOS fluorescence statistics require at least two shots")
        mean = np.zeros(self.calibration.n_sites, dtype=float)
        sum_squared_deviations = np.zeros_like(mean)
        sample_counts = np.zeros(self.calibration.n_sites, dtype=np.int64)
        saturated_sites: set[int] = set()
        missing_sites: set[int] = set()
        measured = 0
        self._active_readout_average = None
        self._active_readout_shots = 0
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
                    image_values = np.asarray(image, dtype=float)
                    if self._active_readout_average is None:
                        self._active_readout_average = np.zeros_like(
                            image_values, dtype=float
                        )
                    self._active_readout_shots += 1
                    self._active_readout_average += (
                        image_values - self._active_readout_average
                    ) / self._active_readout_shots
                    saturated_sites.update(self._saturated_sites(image))
                    counts = self.calibration.signals(
                        image, model_kind=self.model.kind
                    )
                    # Feedback follows the operator-visible repeat mean.  Its
                    # per-site dark level is subtracted, but every valid shot
                    # contributes: loading probability and occupied-atom
                    # fluorescence are both physical functions of trap depth.
                    # A fixed occupied/empty threshold cannot be used here --
                    # the very brightness change being corrected can cross the
                    # calibration threshold and falsely make a real trap look
                    # "missing".
                    sample = (
                        np.asarray(counts, dtype=float)
                        - np.asarray(self.model.dark_mean, dtype=float)
                    )
                    usable = np.isfinite(sample)
                    if np.any(usable):
                        next_counts = sample_counts[usable] + 1
                        delta = sample[usable] - mean[usable]
                        mean[usable] += delta / next_counts
                        sum_squared_deviations[usable] += delta * (
                            sample[usable] - mean[usable]
                        )
                        sample_counts[usable] = next_counts
                    measured += 1
                    context.report_progress(
                        f"Reading mean qCMOS brightness for candidate {iteration + 1}",
                        current=measured,
                        total=requested,
                    )
                wait_for_board(self.sequencer, context)
            finally:
                try:
                    self.sequencer.safe()
                finally:
                    source.close()
        insufficient = sample_counts < 2
        missing_sites.update(int(index) for index in np.flatnonzero(insufficient))
        variance = np.full_like(mean, np.nan)
        enough = ~insufficient
        variance[enough] = (
            sum_squared_deviations[enough] / (sample_counts[enough] - 1)
        )
        standard_error = np.full_like(mean, np.nan)
        standard_error[enough] = np.sqrt(
            variance[enough] / sample_counts[enough]
        )
        missing = np.fromiter(missing_sites, dtype=int)
        if missing.size:
            mean[missing] = np.nan
            standard_error[missing] = np.nan
        if self._active_readout_average is None or self._active_readout_shots != requested:
            raise RuntimeError("qCMOS readout average did not receive every requested shot")
        return (
            mean,
            standard_error,
            tuple(sorted(saturated_sites)),
            tuple(sorted(missing_sites)),
            np.array(self._active_readout_average, copy=True),
        )

    def execute(self, context: object) -> dict[str, object]:
        incoming = np.array(self.slm.last_commanded_phase, copy=True)
        self._generation = str(getattr(context.generation, "value", context.generation))
        self._publication_revision = 0
        self._live_slot = ScanLiveSlot()
        context.attach_live_outputs(self._live_slot)
        history: list[dict[str, object]] = []
        best_valid: dict[str, object] | None = None
        latest_completed: dict[str, object] | None = None
        durable_candidate: dict[str, object] | None = None
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
            accepted: dict[str, object] | None = None
            for iteration in range(self.max_updates):
                _check_cancelled(context)
                applied = self._apply_exact(current_phase)
                candidate_number = iteration + 1
                artifact_path = unique_path(
                    self.artifact_directory,
                    f"slm_feedback_candidate_{candidate_number:04d}",
                    ".npz",
                )
                # The phase file exists before the first camera trigger.  A
                # Stop during that measurement therefore never leaves an
                # unrepeatable phase on the device with no durable source.
                save_phase(
                    artifact_path,
                    applied,
                    self._candidate_metadata(
                        candidate=candidate_number,
                        status="applied",
                        history=history,
                    ),
                )
                durable_candidate = {
                    "candidate": candidate_number,
                    "phase": np.array(applied, copy=True),
                    "artifact_path": artifact_path,
                }
                context.report_progress(
                    f"Candidate {candidate_number} phase saved to {artifact_path}"
                )
                (
                    fluorescence,
                    error,
                    saturated_sites,
                    missing_sites,
                    readout_average,
                ) = self._measure(pulse, context, iteration)
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
                    "iteration": candidate_number,
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
                    "artifact_path": str(artifact_path),
                })
                if valid:
                    coarse_best = min(coarse_best, score)
                save_phase(
                    artifact_path,
                    applied,
                    self._candidate_metadata(
                        candidate=candidate_number,
                        status="measured",
                        history=history,
                    ),
                )
                self._publish_candidate(
                    phase=applied,
                    readout_average=readout_average,
                    artifact_path=artifact_path,
                    candidate=candidate_number,
                    stage="coarse",
                    shots=self.shots,
                    uniformity_ratio=score if valid else None,
                    history=history,
                )
                completed: dict[str, object] = {
                    "candidate": candidate_number,
                    "phase": np.array(applied, copy=True),
                    "readout_average": np.array(readout_average, copy=True),
                    "artifact_path": artifact_path,
                    "stage": "coarse",
                    "shots": self.shots,
                    "uniformity_ratio": score if valid else None,
                    "history": history[-1],
                    "score": score,
                }
                latest_completed = completed
                durable_candidate = completed
                if valid and (
                    best_valid is None or score < float(best_valid["score"])
                ):
                    best_valid = completed
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
                        validation_average,
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
                    save_phase(
                        artifact_path,
                        applied,
                        self._candidate_metadata(
                            candidate=candidate_number,
                            status="validated",
                            history=history,
                        ),
                    )
                    self._publish_candidate(
                        phase=applied,
                        readout_average=validation_average,
                        artifact_path=artifact_path,
                        candidate=candidate_number,
                        stage="validation",
                        shots=self.validation_shots,
                        uniformity_ratio=(
                            validation_score if validation_valid else None
                        ),
                        history=history,
                    )
                    completed = {
                        **completed,
                        "readout_average": np.array(
                            validation_average, copy=True
                        ),
                        "stage": "validation",
                        "shots": self.validation_shots,
                        "uniformity_ratio": (
                            validation_score if validation_valid else None
                        ),
                    }
                    latest_completed = completed
                    durable_candidate = completed
                    if (
                        best_valid is not None
                        and int(best_valid["candidate"]) == candidate_number
                    ):
                        best_valid = completed
                    context.report_progress(
                        f"qCMOS validation ratio {validation_score:.5f}; "
                        f"max relative SEM {max_relative_error:.5f}"
                    )
                    if validation_score <= _TARGET_RATIO:
                        accepted = {
                            **completed,
                            "phase": np.array(applied, copy=True),
                            "history": history[-1],
                            "validation_score": validation_score,
                            "validation_max_relative_standard_error": max_relative_error,
                        }
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
            # A Stop that won keeps the retained candidate through the
            # exception path; after this seal, apply/save lead to success or
            # an exception is a real failure rather than a cancellation.
            context.seal_terminal()
            applied = self._apply_exact(accepted["phase"])
            artifact_path = Path(accepted["artifact_path"])
            final_metadata = self._candidate_metadata(
                candidate=int(accepted["candidate"]),
                status="accepted",
                history=history,
            )
            final_metadata["best"] = accepted["history"]
            save_phase(artifact_path, applied, final_metadata)
            return {
                "artifact_path": artifact_path,
                "best_uniformity": accepted["validation_score"],
                "validation_max_relative_standard_error": accepted[
                    "validation_max_relative_standard_error"
                ],
                "updates": len(history),
            }
        except BaseException as error:
            if context.cancel_requested():
                retained = best_valid if best_valid is not None else latest_completed
                if retained is None:
                    retained = durable_candidate
                    if (
                        retained is not None
                        and self._active_readout_average is not None
                        and self._active_readout_shots > 0
                    ):
                        retained = {
                            **retained,
                            "readout_average": np.array(
                                self._active_readout_average, copy=True
                            ),
                            "stage": "stopped-partial",
                            "shots": self._active_readout_shots,
                            "uniformity_ratio": None,
                        }
                if retained is not None:
                    retained_phase = self._apply_exact(retained["phase"])
                    readout = retained.get("readout_average")
                    if readout is not None:
                        self._publish_candidate(
                            phase=retained_phase,
                            readout_average=np.asarray(readout),
                            artifact_path=Path(retained["artifact_path"]),
                            candidate=int(retained["candidate"]),
                            stage=str(retained.get("stage", "coarse")),
                            shots=int(retained.get("shots", self.shots)),
                            uniformity_ratio=retained.get("uniformity_ratio"),
                            history=history,
                        )
                    raise
            try:
                self._apply_exact(incoming)
            except BaseException as restore_error:
                raise RuntimeError(
                    f"SLM feedback failed and the incoming phase could not be restored: {error}"
                ) from restore_error
            raise


__all__ = [
    "CANDIDATE_PHASE_OUTPUT",
    "READOUT_AVERAGE_OUTPUT",
    "SLM_PHASE_ARTIFACT_CONTRACT",
    "SlmFeedbackTask",
    "UNIFORMITY_HISTORY_OUTPUT",
]
