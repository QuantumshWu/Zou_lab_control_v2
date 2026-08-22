"""Repeat-mean site-brightness feedback; no hidden-plant inference."""

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
)
from zlc_atom.nodes.calibration.calibration import (
    _register_target_sites,
    reads_photoelectrons,
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
_CENSORED_BOOST_LOG_STEP = float(np.log(2.0))
_COARSE_MAX_BATCHES = 3
_MAX_BOOTSTRAP_UPDATES = 3
_VALIDATION_BATCH_SHOTS = 100
_VALIDATION_MAX_SECONDS = 300.0
READOUT_FRAME_COORDINATE = 1


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
        raise ValueError("fluorescence estimate and uncertainty must be finite and positive")
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


def _censored_sites(
    values: np.ndarray,
    standard_error: np.ndarray,
    *,
    looks: int = 1,
) -> tuple[int, ...]:
    measured = np.asarray(values, dtype=float)
    error = np.asarray(standard_error, dtype=float)
    if (
        measured.ndim != 1
        or error.shape != measured.shape
        or type(looks) is not int
        or looks < 1
    ):
        raise ValueError("fluorescence estimate and uncertainty shapes differ")
    invalid = ~np.isfinite(measured) | ~np.isfinite(error) | (error < 0.0)
    z = float(
        special.ndtri(1.0 - 0.05 / (2.0 * len(measured) * int(looks)))
    )
    return tuple(
        int(index)
        for index in np.flatnonzero(invalid | (measured - z * error <= 0.0))
    )


def _boost_target(
    target: np.ndarray,
    censored_sites: tuple[int, ...],
    rows: np.ndarray,
    columns: np.ndarray,
) -> np.ndarray:
    updated = np.array(target, dtype=np.float32, copy=True)
    indices = np.asarray(censored_sites, dtype=int)
    updated[rows[indices], columns[indices]] *= np.float32(
        np.exp(_CENSORED_BOOST_LOG_STEP)
    )
    updated *= float(np.sum(target)) / float(np.sum(updated))
    return validate_target(updated)


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
    np.ndarray,
    np.ndarray,
]:
    """Register a generic camera calibration to this Feedback Target."""

    model = calibration.select_model(ReadoutModelKind.BOX)
    dark_mean = np.asarray(model.dark_mean, dtype=float)
    dark_count = np.asarray(model.dark_sample_count)
    dark_variance = np.asarray(model.dark_sample_variance, dtype=float)
    usable = (
        np.asarray(calibration.site_map.valid_sites, dtype=bool)
        & np.isfinite(dark_mean)
        & (dark_count >= 2)
        & np.isfinite(dark_variance)
        & (dark_variance >= 0.0)
    )
    if not np.any(usable):
        raise ValueError(
            "SLM Feedback requires at least one calibrated BOX site with dark statistics"
        )
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

    observed = np.asarray(registered.topology["observed_sites"], dtype=bool)
    centers = np.asarray(registered.centers_xy, dtype=float)
    roster_dark = np.full(len(centers), np.nan, dtype=float)
    roster_sem_squared = np.full(len(centers), np.nan, dtype=float)
    used_sources: set[int] = set()
    for roster_index in np.flatnonzero(observed):
        distance = np.linalg.norm(
            np.asarray(source_map.centers_xy, dtype=float) - centers[roster_index],
            axis=1,
        )
        local_index = int(np.argmin(distance))
        if distance[local_index] > 1e-9 or local_index in used_sources:
            raise RuntimeError("Feedback registration lost a measured Calibration site")
        used_sources.add(local_index)
        source_index = int(source_indices[local_index])
        roster_dark[roster_index] = dark_mean[source_index]
        roster_sem_squared[roster_index] = (
            dark_variance[source_index] / dark_count[source_index]
        )

    missing = np.flatnonzero(~observed)
    if len(missing):
        observed_indices = np.flatnonzero(observed)
        observed_dark = roster_dark[observed_indices]
        spatial_scale = 1.4826 * float(
            np.median(np.abs(observed_dark - np.median(observed_dark)))
        )
        systematic_variance = max(
            spatial_scale * spatial_scale,
            float(np.max(roster_sem_squared[observed_indices])),
        )
        for roster_index in missing:
            nearest = int(
                observed_indices[
                    np.argmin(
                        np.linalg.norm(
                            centers[observed_indices] - centers[roster_index], axis=1
                        )
                    )
                ]
            )
            roster_dark[roster_index] = roster_dark[nearest]
            roster_sem_squared[roster_index] = (
                roster_sem_squared[nearest] + systematic_variance
            )
    if not np.all(np.isfinite(roster_dark)) or not np.all(
        np.isfinite(roster_sem_squared)
    ):
        raise RuntimeError("Feedback registration produced invalid dark statistics")
    return rows, columns, centers, roster_dark, roster_sem_squared


def _updated_target(
    target: np.ndarray,
    fluorescence: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
) -> np.ndarray:
    values = np.asarray(fluorescence, dtype=float)
    if (
        values.shape != (len(rows),)
        or not np.all(np.isfinite(values))
        or np.any(values <= 0.0)
    ):
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
        if not isinstance(receipt, Mapping):
            raise TypeError("Science Context command receipt must be a mapping")
        model = calibration.select_model(ReadoutModelKind.BOX)
        context_path = Path(science_context_path).expanduser().resolve()
        (
            self._rows,
            self._columns,
            self._site_centers_xy,
            self._dark_mean,
            self._dark_sem_squared,
        ) = _support(
            frozen_target,
            calibration,
            science_context_path=context_path,
            command_receipt=receipt,
        )
        self._site_count = len(self._rows)
        self._site_centers_xy.setflags(write=False)
        self._dark_mean.setflags(write=False)
        self._dark_sem_squared.setflags(write=False)
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
        self.shots = int(shots_per_candidate)
        self.validation_shots = int(validation_shots)
        self.max_updates = int(max_updates)
        self._candidate_capacity = self.max_updates + 1
        self._publication_revision = 0
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
            expected_photoelectrons = reads_photoelectrons(self.calibration)
            if node.reads_photoelectrons != expected_photoelectrons:
                raise ValueError(
                    "camera effective photoelectron mode differs from Calibration"
                )
            raw_dtype = np.dtype(actual.dtype)
            if raw_dtype.kind not in "iu":
                raise ValueError(
                    "Feedback saturation requires an integer raw camera dtype"
                )
            if not isinstance(actual.count_unit, str) or not actual.count_unit:
                raise ValueError("Feedback camera count_unit is invalid")
            run_record = self.calibration.report.get("run_record")
            recorded_camera = None
            if isinstance(run_record, Mapping):
                devices = run_record.get("actual_devices")
                if isinstance(devices, Mapping):
                    recorded_camera = devices.get(self.camera_key)
            if not isinstance(recorded_camera, Mapping):
                raise ValueError("Calibration lacks camera working-point provenance")
            if (
                recorded_camera.get("dtype") != raw_dtype.str
                or recorded_camera.get("count_unit") != actual.count_unit
            ):
                raise ValueError("camera raw dtype/count unit differs from Calibration")
            raw_maximum = np.iinfo(raw_dtype).max
            if expected_photoelectrons:
                recorded_offset = recorded_camera.get("offset_counts")
                recorded_scale = recorded_camera.get("electrons_per_count")
                current_offset = actual.offset_counts
                current_scale = actual.electrons_per_count
                if (
                    type(recorded_offset) not in (int, float)
                    or type(recorded_scale) not in (int, float)
                    or current_offset is None
                    or current_scale is None
                    or not np.isfinite(recorded_offset)
                    or not np.isfinite(recorded_scale)
                    or recorded_scale <= 0.0
                    or float(recorded_offset) != float(current_offset)
                    or float(recorded_scale) != float(current_scale)
                ):
                    raise ValueError(
                        "camera photoelectron conversion differs from Calibration"
                    )
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
        mean = np.full(self._site_count, np.nan, dtype=float)
        mean[complete] = (
            np.mean(box_samples[:, complete], axis=0) - self._dark_mean[complete]
        )
        variance = np.full(self._site_count, np.nan, dtype=float)
        variance[complete] = np.var(box_samples[:, complete], axis=0, ddof=1)
        standard_error = np.full_like(mean, np.nan)
        standard_error[complete] = np.sqrt(
            variance[complete] / requested
            + self._dark_sem_squared[complete]
        )
        return (
            mean,
            standard_error,
            tuple(sorted(saturated_sites)),
            tuple(sorted(missing_sites)),
        )

    def _coarse_measure(
        self,
        pulse: object,
        context: object,
        iteration: int,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        tuple[int, ...],
        tuple[int, ...],
        tuple[int, ...],
        int,
        int,
    ]:
        for attempt in range(2):
            mean = np.zeros(self._site_count, dtype=float)
            m2 = np.zeros_like(mean)
            error = np.full_like(mean, np.nan)
            count = 0
            saturated: tuple[int, ...] = ()
            missing: tuple[int, ...] = ()
            censored = tuple(range(self._site_count))
            for batch in range(_COARSE_MAX_BATCHES):
                batch_mean, batch_error, saturated, missing = self._measure(
                    pulse,
                    context,
                    iteration,
                    shots=None if batch == 0 else self.shots,
                )
                finite = (
                    np.all(np.isfinite(batch_mean))
                    and np.all(np.isfinite(batch_error))
                    and np.all(np.asarray(batch_error) >= 0.0)
                )
                if saturated or missing or not finite:
                    mean = np.array(batch_mean, dtype=float, copy=True)
                    error = np.array(batch_error, dtype=float, copy=True)
                    if not missing:
                        missing = tuple(
                            int(index)
                            for index in np.flatnonzero(
                                ~np.isfinite(batch_mean)
                                | ~np.isfinite(batch_error)
                            )
                        )
                    break
                batch_m2 = (
                    np.maximum(
                        np.square(batch_error) - self._dark_sem_squared,
                        0.0,
                    )
                    * self.shots
                    * (self.shots - 1)
                )
                combined = count + self.shots
                delta = batch_mean - mean
                mean += delta * self.shots / combined
                m2 += (
                    batch_m2
                    + np.square(delta) * count * self.shots / combined
                )
                count = combined
                error = np.sqrt(
                    m2 / (count - 1) / count + self._dark_sem_squared
                )
                censored = _censored_sites(
                    mean, error, looks=_COARSE_MAX_BATCHES
                )
                if not censored:
                    return (
                        mean,
                        error,
                        (),
                        (),
                        (),
                        attempt + 1,
                        count,
                    )
            if not saturated and not missing and count:
                return (
                    mean,
                    np.sqrt(
                        m2 / (count - 1) / count + self._dark_sem_squared
                    ),
                    (),
                    (),
                    censored,
                    attempt + 1,
                    count,
                )
            if attempt == 0:
                context.report_progress(
                    f"Candidate {iteration + 1} fluorescence invalid; "
                    "retrying the same applied phase"
                )
        return (
            mean,
            error,
            saturated,
            missing,
            (),
            2,
            0,
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
        }

    def execute(self, context: object) -> dict[str, object]:
        incoming = self._incoming_phase
        incoming_pattern = self._pattern_phase
        history: list[dict[str, object]] = []
        retained_valid: dict[str, object] | None = None
        best_observed: dict[str, object] | None = None
        try:
            _check_cancelled(context)
            # Science Context is the requested starting CONTENT, not proof of
            # what a previous process happens to have commanded. This Task owns
            # the SLM now, so establish and confirm that starting state itself.
            self._mapping_revision = int(self.slm.mapping_revision)
            incoming = self._apply_exact(incoming)
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
            _check_cancelled(context)
            current_target = self.target
            spot_optimizer_state: dict[str, object] = {}
            current_pattern = incoming_pattern
            current_phase = incoming
            solver_metadata: Mapping[str, object] | None = None
            bootstrap_updates = 0
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
                    fluorescence,
                    error,
                    saturated_sites,
                    missing_sites,
                    censored_sites,
                    attempts,
                    coarse_shots,
                ) = self._coarse_measure(pulse, context, iteration)
                saturated = bool(saturated_sites)
                missing = bool(missing_sites)
                valid = bool(not saturated and not missing and not censored_sites)
                observed_score = (
                    float(np.max(fluorescence) / np.min(fluorescence))
                    if np.all(np.isfinite(fluorescence))
                    and np.all(fluorescence > 0.0)
                    else float("nan")
                )
                if valid:
                    score, confidence_lower, confidence_upper, relative_sem = (
                        _ratio_interval(
                            fluorescence,
                            error,
                            looks=_COARSE_MAX_BATCHES,
                        )
                    )
                else:
                    score = confidence_lower = confidence_upper = relative_sem = float("inf")
                visibility = self._site_count - len(censored_sites)
                visibility_margin = (
                    None
                    if saturated or missing
                    else float(
                        np.min(
                            fluorescence
                            - float(
                                special.ndtri(
                                    1.0
                                    - 0.05
                                    / (
                                        2.0
                                        * self._site_count
                                        * _COARSE_MAX_BATCHES
                                    )
                                )
                            )
                            * error
                        )
                    )
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
                    "bootstrap_updates": bootstrap_updates,
                    "fluorescence": _json_floats(fluorescence),
                    "standard_error": _json_floats(error),
                    "missing_sites": list(missing_sites),
                    "censored_sites": list(censored_sites),
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
                    "fluorescence": np.array(fluorescence, copy=True),
                    "standard_error": np.array(error, copy=True),
                }
                if saturated or missing:
                    raise RuntimeError(
                        "qCMOS fluorescence remained invalid after two measurements "
                        "of the same candidate"
                    )
                completed["visibility_rank"] = (visibility, visibility_margin)
                if (
                    best_observed is None
                    or completed["visibility_rank"]
                    > best_observed["visibility_rank"]
                ):
                    best_observed = completed
                if not valid:
                    if retained_valid is not None:
                        history[-1]["rollback_to_candidate"] = int(
                            retained_valid["candidate"]
                        )
                        self._apply_exact(retained_valid["phase"])
                        context.report_progress(
                            f"Candidate {candidate_number} was censored at "
                            f"{len(censored_sites)} site(s); retaining candidate "
                            f"{retained_valid['candidate']}"
                        )
                        break
                    context.report_progress(
                        f"Candidate {candidate_number} was censored at "
                        f"{len(censored_sites)} site(s) after {coarse_shots} shots; "
                        "applying one bounded bootstrap step"
                    )
                    if (
                        candidate_number == self._candidate_capacity
                        or bootstrap_updates >= _MAX_BOOTSTRAP_UPDATES
                    ):
                        break
                    current_target = _boost_target(
                        current_target,
                        censored_sites,
                        self._rows,
                        self._columns,
                    )
                    bootstrap_updates += 1
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
                    continue
                retained_valid = completed
                context.report_progress(
                    f"qCMOS fluorescence ratio {score:.5f}; "
                    f"simultaneous 95% upper {confidence_upper:.5f}"
                )
                if score <= _TARGET_RATIO:
                    break
                if candidate_number == self._candidate_capacity:
                    break
                if valid:
                    current_target = _updated_target(
                        current_target,
                        fluorescence,
                        self._rows,
                        self._columns,
                    )
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
                if best_observed is None:
                    raise RuntimeError("qCMOS feedback produced no observable candidate")
                best_observed["history"]["validation"] = {
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
                    "fluorescence": _json_floats(best_observed["fluorescence"]),
                    "standard_error": _json_floats(
                        best_observed["standard_error"]
                    ),
                    "censored_sites": best_observed["history"][
                        "censored_sites"
                    ],
                }
                inconclusive = {
                    **self._incoming_candidate(
                        stem="slm_feedback_inconclusive",
                        status="inconclusive",
                        history=history,
                        observed=best_observed,
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
            validation_mean = np.zeros(self._site_count, dtype=float)
            validation_m2 = np.zeros_like(validation_mean)
            validation_count = 0
            validation_error = np.full_like(validation_mean, np.nan)
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
                    batch_mean,
                    batch_error,
                    saturated_sites,
                    missing_sites,
                ) = self._measure(
                    pulse,
                    context,
                    int(retained_valid["candidate"]) - 1,
                    shots=batch_shots,
                )
                batch_valid = bool(
                    not saturated_sites
                    and not missing_sites
                    and np.all(np.isfinite(batch_mean))
                    and np.all(np.isfinite(batch_error))
                    and np.all(batch_error >= 0.0)
                )
                if not batch_valid:
                    validation_reason = "independent validation data were invalid"
                    break
                batch_m2 = (
                    np.maximum(
                        np.square(batch_error) - self._dark_sem_squared,
                        0.0,
                    )
                    * batch_shots
                    * (batch_shots - 1)
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
                        + self._dark_sem_squared
                    )
                    validation_censored = _censored_sites(
                        validation_mean,
                        validation_error,
                        looks=validation_max_looks,
                    )
                    if validation_censored:
                        validation_reason = (
                            "independent validation sites remained censored"
                        )
                        context.report_progress(
                            f"Independent validation {validation_count}/"
                            f"{self.validation_shots}: "
                            f"{len(validation_censored)} censored site(s)"
                        )
                        continue
                    (
                        validation_estimate,
                        validation_lower,
                        validation_upper,
                        validation_relative_sem,
                    ) = _ratio_interval(
                        validation_mean,
                        validation_error,
                        looks=validation_max_looks,
                    )
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
                "fluorescence": _json_floats(validation_mean),
                "standard_error": _json_floats(validation_error),
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
                            observed=best_observed,
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
