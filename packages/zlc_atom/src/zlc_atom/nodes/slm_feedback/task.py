"""Repeat-mean site-brightness feedback; no hidden-plant inference."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Mapping

import numpy as np
from scipy import special
from scipy.optimize import linear_sum_assignment
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
    save_science_context,
    solve_phase,
    validate_target,
)
from zlc_atom.nodes.calibration import ReadoutModelKind, TrapCalibration
from zlc_atom.nodes.calibration.calibration import reads_photoelectrons
from zlc_atom.nodes.calibration.pulse import arm_sequencer, resolve_pulse
from zlc_atom.nodes.camera_measurement.measurement import (
    CameraMeasurementNode,
    CameraMeasurementRequest,
)
from zlc_atom.nodes.scan.source import wait_for_board


SLM_PHASE_ARTIFACT_CONTRACT = "zlc.slm.science-context.v1"
CANDIDATE_PHASE_OUTPUT = DatasetOutputDeclaration(
    "candidate_phase", "slm-feedback.candidate-phase.v1"
)
UNIFORMITY_HISTORY_OUTPUT = DatasetOutputDeclaration(
    "uniformity_history", "slm-feedback.uniformity-history.v1"
)
_TARGET_RATIO = 1.01
_FEEDBACK_EXPONENT = 0.25
_MAX_LOG_STEP = 0.2
_MIN_FEEDBACK_GAIN = 0.03
_MAX_FEEDBACK_GAIN = 0.35
_NO_IMPROVEMENT_PATIENCE = 3
_VALIDATION_BATCH_SHOTS = 100
_VALIDATION_MAX_SECONDS = 60.0
READOUT_FRAME_COORDINATE = 1
_GEOMETRY_TOLERANCE_FRACTION = 0.25
_MAX_NORMALIZED_AFFINE_DEVIATION = 0.25
_MAX_NORMALIZED_AFFINE_CONDITION = 3.0
_MAX_CROSS_AXIS_SPAN_FRACTION = 0.25


def _check_cancelled(context: object) -> None:
    if context.cancel_requested():
        raise RuntimeError("SLM feedback was cancelled")


def _readout_frames(snapshot: object, *, shots: int) -> np.ndarray:
    """Apply the preview's frame=1 scope to one sealed Camera dataset."""

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
    values: np.ndarray, standard_error: np.ndarray
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
        raise ValueError("fluorescence estimate and uncertainty must be finite and positive")
    relative = error / measured
    z = float(special.ndtri(1.0 - 0.05 / (2.0 * len(measured))))
    logarithm = np.log(measured)
    estimate = float(np.exp(np.max(logarithm) - np.min(logarithm)))
    lower = float(
        np.exp(np.max(logarithm - z * relative) - np.min(logarithm + z * relative))
    )
    upper = float(
        np.exp(np.max(logarithm + z * relative) - np.min(logarithm - z * relative))
    )
    return estimate, max(1.0, lower), upper, float(np.max(relative))


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


def _updated_target(
    target: np.ndarray,
    fluorescence: np.ndarray,
    standard_error: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    *,
    gain: float,
) -> np.ndarray:
    values = np.asarray(fluorescence, dtype=float)
    error = np.asarray(standard_error, dtype=float)
    if (
        values.shape != (len(rows),)
        or error.shape != values.shape
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(error))
        or np.any(values <= 0.0)
        or np.any(error < 0.0)
        or not np.isfinite(gain)
        or gain <= 0.0
    ):
        raise ValueError("feedback fluorescence must be finite and positive at every site")
    logarithm = np.log(values)
    residual = logarithm - float(np.mean(logarithm))
    z = float(special.ndtri(1.0 - 0.05 / (2.0 * len(values))))
    resolved = np.sign(residual) * np.maximum(
        np.abs(residual) - z * error / values, 0.0
    )
    step = np.clip(-float(gain) * resolved, -_MAX_LOG_STEP, _MAX_LOG_STEP)
    updated = np.array(target, dtype=np.float32, copy=True)
    updated[rows, columns] *= np.asarray(np.exp(step), dtype=np.float32)
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
        target_objective: str,
        target_path: str | Path,
        science_context: Mapping[str, object],
        science_context_path: str | Path,
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
        if target_objective != "spots":
            raise ValueError("SLM fluorescence feedback accepts sparse spot targets only")
        if frozen_target.shape != slm.shape_yx:
            raise ValueError("target shape differs from the selected SLM")
        if not isinstance(science_context, Mapping):
            raise TypeError("science_context must be a loaded Science Context mapping")
        if science_context.get("objective_kind") != "spots":
            raise ValueError("SLM feedback Science Context must use the spots objective")
        incoming = canonical_phase(science_context.get("phase"), slm.shape_yx)
        pattern = canonical_phase(
            science_context.get("pattern_phase"), slm.shape_yx
        )
        operator = canonical_phase(
            science_context.get("operator_wavefront"), slm.shape_yx
        )
        composed = canonical_phase(
            pattern.astype(float) + operator.astype(float), slm.shape_yx
        )
        if not np.array_equal(incoming, composed):
            raise ValueError("Science Context phase differs from Pattern plus wavefront")
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
        if not isinstance(receipt, Mapping) or receipt.get("outcome") not in {
            "known-old",
            "known-new",
        }:
            raise ValueError("SLM feedback requires a known incoming command receipt")
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
        self.science_context_path = Path(science_context_path).expanduser().resolve()
        self._incoming_phase = incoming
        self._pattern_phase = pattern
        self._operator_wavefront = operator
        self._pupil_amplitude = np.array(pupil, copy=True)
        self._pupil_amplitude.setflags(write=False)
        self._pupil_support = np.array(support, copy=True)
        self._pupil_support.setflags(write=False)
        self._pupil = dict(science_context.get("pupil", {}))
        self._system_correction = science_context.get("system_correction")
        self._incoming_receipt = dict(receipt)
        self._pattern_metadata = dict(science_context.get("pattern_metadata", {}))
        self._operator_metadata = dict(science_context.get("operator_metadata", {}))
        self._mapping_revision = int(receipt["mapping_revision"])
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

    @property
    def dataset_output_declarations(self) -> tuple[DatasetOutputDeclaration, ...]:
        return (
            CANDIDATE_PHASE_OUTPUT,
            UNIFORMITY_HISTORY_OUTPUT,
        )

    def _run_record(self) -> dict[str, object]:
        return {
            "calibration_path": str(self.calibration_path),
            "target_path": str(self.target_path),
            "science_context_path": str(self.science_context_path),
            "pulse_path": str(self.pulse_path),
            "named_devices": {
                "camera": self.camera_key,
                "sequencer": self.sequencer_key,
                "slm": self.slm_key,
            },
            "max_updates": self.max_updates,
        }

    def _candidate_metadata(
        self,
        *,
        candidate: int,
        status: str,
        history: list[dict[str, object]],
        target: object,
        solver: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        return {
            "calibration_path": str(self.calibration_path),
            "target_path": str(self.target_path),
            "science_context_path": str(self.science_context_path),
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
            "updates": len(history),
            "target_support_yx": np.column_stack(
                (self._rows, self._columns)
            ).astype(int).tolist(),
            "target_site_intensity": np.asarray(target)[
                self._rows, self._columns
            ].astype(float).tolist(),
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
        if not 1 <= candidate <= self.max_updates:
            raise ValueError("feedback candidate lies outside its authored history")
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
            revision=candidate,
        )
        coordinate_id = AxisId("slm_feedback.candidate")
        curve = np.full(self.max_updates, np.nan, dtype="<f8")
        for item in history:
            ratio = item.get("uniformity_ratio")
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
                        for index in range(1, self.max_updates + 1)
                    ),
                )
            },
            generation=generation,
            revision=candidate,
            validity=np.isfinite(curve)[None],
        )
        context.commit_live(
            {
                CANDIDATE_PHASE_OUTPUT.name: LiveDatasetOutput(
                    CANDIDATE_PHASE_OUTPUT,
                    phase_event,
                    MonitorCoverage(1, 1),
                    record,
                ),
                UNIFORMITY_HISTORY_OUTPUT.name: LiveDatasetOutput(
                    UNIFORMITY_HISTORY_OUTPUT,
                    history_event,
                    MonitorCoverage(self.max_updates, self.max_updates),
                    record,
                ),
            }
        )

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
    ]:
        contract = self.calibration.frame_contract
        if contract.exposure_seconds is None:
            raise ValueError("calibration does not record its readout exposure")
        requested = self.shots if shots is None else int(shots)
        if requested < 2:
            raise ValueError("qCMOS fluorescence statistics require at least two shots")
        camera_owner = f"{context.instance_id}/camera"
        node = CameraMeasurementNode(
            camera=self.camera,
            request=CameraMeasurementRequest(
                camera_key=self.camera_key,
                exposure_seconds=float(contract.exposure_seconds),
                roi_xywh=contract.roi_xywh,
                repeat=requested,
                frames_per_cycle=3,
                photoelectrons=reads_photoelectrons(self.calibration),
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

        mean = np.zeros(self.calibration.n_sites, dtype=float)
        sum_squared_deviations = np.zeros_like(mean)
        sample_counts = np.zeros(self.calibration.n_sites, dtype=np.int64)
        saturated_sites: set[int] = set()
        for image in frames:
            saturated_sites.update(self._saturated_sites(image))
            counts = self.calibration.signals(image, model_kind=self.model.kind)
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
        complete = sample_counts == requested
        missing_sites = set(int(index) for index in np.flatnonzero(~complete))
        variance = np.full_like(mean, np.nan)
        variance[complete] = (
            sum_squared_deviations[complete] / (sample_counts[complete] - 1)
        )
        standard_error = np.full_like(mean, np.nan)
        standard_error[complete] = np.sqrt(
            variance[complete] / sample_counts[complete]
        )
        if missing_sites:
            missing = np.fromiter(missing_sites, dtype=int)
            mean[missing] = np.nan
            standard_error[missing] = np.nan
        return (
            mean,
            standard_error,
            tuple(sorted(saturated_sites)),
            tuple(sorted(missing_sites)),
        )

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
            candidate=candidate_number,
            status=status,
            history=history,
            target=candidate["target"],
            solver=candidate.get("solver"),
        )
        metadata["best"] = candidate.get("history")
        metadata["history"] = history
        self._save_candidate(
            artifact_path,
            applied,
            candidate["pattern_phase"],
            metadata,
        )
        best = candidate.get("validation_score")
        if best is None and isinstance(candidate.get("history"), Mapping):
            best = candidate["history"].get("uniformity_ratio")
        return {
            "artifact_path": artifact_path,
            "best_uniformity": best,
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
        }

    def execute(self, context: object) -> dict[str, object]:
        observed_incoming = self.slm.last_commanded_phase
        observed_receipt = dict(self.slm.last_command_receipt)
        if (
            observed_incoming is None
            or not np.array_equal(observed_incoming, self._incoming_phase)
            or json.dumps(
                observed_receipt,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            != json.dumps(
                self._incoming_receipt,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            or self.slm.command_revision
            != self._incoming_receipt.get("command_revision")
            or self.slm.mapping_revision != self._mapping_revision
        ):
            raise RuntimeError(
                "SLM command no longer matches the frozen incoming Science Context"
            )
        incoming = self._incoming_phase
        incoming_pattern = self._pattern_phase
        history: list[dict[str, object]] = []
        best_valid: dict[str, object] | None = None
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
            spot_optimizer_state: dict[str, object] = {}
            current_pattern, solver_metadata = solve_phase(
                current_target,
                pupil_amplitude=self._pupil_amplitude,
                initial_phase=incoming_pattern,
                objective_kind="spots",
                iterations=None,
                stop_requested=context.cancel_requested,
                spot_optimizer_state=spot_optimizer_state,
            )
            current_phase = self._science_phase(current_pattern)
            gain = _FEEDBACK_EXPONENT
            no_improvement = 0
            for iteration in range(self.max_updates):
                _check_cancelled(context)
                applied = self._apply_exact(current_phase)
                candidate_number = iteration + 1
                applied_metadata = self._candidate_metadata(
                    candidate=candidate_number,
                    status="applied",
                    history=history,
                    target=current_target,
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
                        applied_metadata,
                    ),
                )
                context.report_progress(
                    f"Candidate {candidate_number} phase saved to {artifact_path}"
                )
                for attempt in range(2):
                    (
                        fluorescence,
                        error,
                        saturated_sites,
                        missing_sites,
                    ) = self._measure(pulse, context, iteration)
                    saturated = bool(saturated_sites)
                    missing = bool(missing_sites)
                    valid = bool(
                        not saturated
                        and not missing
                        and np.all(np.isfinite(fluorescence))
                        and np.all(np.isfinite(error))
                        and np.all(fluorescence > 0.0)
                        and np.all(error >= 0.0)
                    )
                    if valid:
                        break
                    if attempt == 0:
                        context.report_progress(
                            f"Candidate {candidate_number} fluorescence invalid; "
                            "retrying the same applied phase"
                        )
                if valid:
                    score, confidence_lower, confidence_upper, relative_sem = (
                        _ratio_interval(fluorescence, error)
                    )
                else:
                    score = confidence_lower = confidence_upper = relative_sem = float("inf")
                history.append({
                    "iteration": candidate_number,
                    "shots": self.shots,
                    "attempts": attempt + 1,
                    "valid": valid,
                    "saturated": saturated,
                    "saturated_sites": list(saturated_sites),
                    "uniformity_ratio": None if not valid else score,
                    "uniformity_confidence_lower": (
                        None if not valid else confidence_lower
                    ),
                    "uniformity_confidence_upper": (
                        None if not valid else confidence_upper
                    ),
                    "maximum_relative_standard_error": (
                        None if not valid else relative_sem
                    ),
                    "controller_gain": gain,
                    "fluorescence": _json_floats(fluorescence),
                    "standard_error": _json_floats(error),
                    "missing_sites": list(missing_sites),
                    "artifact_path": str(artifact_path),
                })
                self._save_candidate(
                    artifact_path,
                    applied,
                    current_pattern,
                    self._candidate_metadata(
                        candidate=candidate_number,
                        status="measured",
                        history=history,
                        target=current_target,
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
                    "score": confidence_upper,
                    "fluorescence": np.array(fluorescence, copy=True),
                    "standard_error": np.array(error, copy=True),
                }
                if not valid:
                    raise RuntimeError(
                        "qCMOS fluorescence remained invalid after two measurements "
                        "of the same candidate"
                    )
                improved = (
                    best_valid is None
                    or confidence_upper < float(best_valid["score"])
                )
                if improved:
                    had_best = best_valid is not None
                    best_valid = completed
                    no_improvement = 0
                    if had_best:
                        gain = min(_MAX_FEEDBACK_GAIN, gain * 1.1)
                else:
                    no_improvement += 1
                    gain = max(_MIN_FEEDBACK_GAIN, gain * 0.5)
                    history[-1]["rollback_to_candidate"] = int(
                        best_valid["candidate"]
                    )
                    self._apply_exact(best_valid["phase"])
                context.report_progress(
                    f"qCMOS fluorescence ratio {score:.5f}; "
                    f"simultaneous 95% upper {confidence_upper:.5f}; "
                    f"gain {gain:.3f}"
                )
                if score <= _TARGET_RATIO or no_improvement >= _NO_IMPROVEMENT_PATIENCE:
                    break
                if candidate_number == self.max_updates:
                    break
                if valid:
                    current_target = _updated_target(
                        best_valid["target"],
                        best_valid["fluorescence"],
                        best_valid["standard_error"],
                        self._rows,
                        self._columns,
                        gain=gain,
                    )
                    current_pattern, solver_metadata = solve_phase(
                        current_target,
                        pupil_amplitude=self._pupil_amplitude,
                        initial_phase=best_valid["pattern_phase"],
                        objective_kind="spots",
                        iterations=None,
                        stop_requested=context.cancel_requested,
                        spot_optimizer_state=spot_optimizer_state,
                    )
                    current_phase = self._science_phase(current_pattern)
            if best_valid is None:
                raise RuntimeError("qCMOS feedback produced no valid coarse candidate")
            _check_cancelled(context)
            self._apply_exact(best_valid["phase"])
            validation_mean = np.zeros(self.calibration.n_sites, dtype=float)
            validation_m2 = np.zeros_like(validation_mean)
            validation_count = 0
            validation_error = np.full_like(validation_mean, np.nan)
            validation_estimate = validation_lower = validation_upper = float("inf")
            validation_relative_sem = float("inf")
            validation_status = "inconclusive"
            validation_reason = "maximum validation shots reached"
            validation_started = time.monotonic()
            deadline = validation_started + _VALIDATION_MAX_SECONDS
            while (
                validation_count < self.validation_shots
                and time.monotonic() < deadline
            ):
                _check_cancelled(context)
                batch_shots = min(
                    _VALIDATION_BATCH_SHOTS,
                    self.validation_shots - validation_count,
                )
                (
                    batch_mean,
                    batch_error,
                    saturated_sites,
                    missing_sites,
                ) = self._measure(
                    pulse,
                    context,
                    int(best_valid["candidate"]) - 1,
                    shots=batch_shots,
                )
                batch_valid = bool(
                    not saturated_sites
                    and not missing_sites
                    and np.all(np.isfinite(batch_mean))
                    and np.all(np.isfinite(batch_error))
                    and np.all(batch_mean > 0.0)
                    and np.all(batch_error >= 0.0)
                )
                if not batch_valid:
                    validation_reason = "independent validation data were invalid"
                    break
                batch_m2 = (
                    np.square(batch_error) * batch_shots * (batch_shots - 1)
                )
                combined_count = validation_count + batch_shots
                delta = batch_mean - validation_mean
                validation_mean += delta * batch_shots / combined_count
                validation_m2 += (
                    batch_m2
                    + np.square(delta)
                    * validation_count
                    * batch_shots
                    / combined_count
                )
                validation_count = combined_count
                if validation_count >= 2:
                    validation_error = np.sqrt(
                        validation_m2
                        / (validation_count - 1)
                        / validation_count
                    )
                    (
                        validation_estimate,
                        validation_lower,
                        validation_upper,
                        validation_relative_sem,
                    ) = _ratio_interval(validation_mean, validation_error)
                    context.report_progress(
                        f"Independent validation {validation_count}/"
                        f"{self.validation_shots}: ratio {validation_estimate:.5f}, "
                        f"95% upper {validation_upper:.5f}"
                    )
                    if validation_upper <= _TARGET_RATIO:
                        validation_status = "accepted"
                        validation_reason = "simultaneous confidence bound passed"
                        break
                    if validation_lower > _TARGET_RATIO:
                        validation_reason = "simultaneous confidence bound excludes 1.01"
                        break
            if time.monotonic() >= deadline and validation_status != "accepted":
                validation_reason = "validation time budget reached"
            best_valid["history"]["validation"] = {
                "status": validation_status,
                "reason": validation_reason,
                "shots": validation_count,
                "maximum_shots": self.validation_shots,
                "maximum_seconds": _VALIDATION_MAX_SECONDS,
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
                "fluorescence": _json_floats(validation_mean),
                "standard_error": _json_floats(validation_error),
            }
            accepted = {
                **best_valid,
                "history": best_valid["history"],
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
                republish=False,
            )
        except BaseException as error:
            if context.cancel_requested():
                try:
                    context.seal_terminal(accept_stop=True)
                    retained = best_valid
                    if retained is None:
                        candidate_number = max(1, len(history) + 1)
                        metadata = self._candidate_metadata(
                            candidate=candidate_number,
                            status="stopped-before-measurement",
                            history=history,
                            target=self.target,
                        )
                        artifact_path = unique_path(
                            self.artifact_directory,
                            "slm_feedback_stopped",
                            ".npz",
                            writer=lambda temporary: self._save_candidate(
                                temporary,
                                incoming,
                                incoming_pattern,
                                metadata,
                            ),
                        )
                        retained = {
                            "candidate": candidate_number,
                            "phase": np.array(incoming, copy=True),
                            "pattern_phase": np.array(incoming_pattern, copy=True),
                            "artifact_path": artifact_path,
                            "target": np.array(self.target, copy=True),
                            "solver": None,
                            "uniformity_ratio": None,
                            "history": None,
                        }
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
