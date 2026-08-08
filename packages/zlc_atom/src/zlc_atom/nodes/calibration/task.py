"""Artifact-only calibration orchestration over camera and sequencer adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from zlc_durable import unique_path

from zlc_atom.devices.camera.contract import (
    CameraAdapter,
    CameraCaptureTerminalRecord,
    CameraFrameRecord,
    CameraWorkingPoint,
)
from .pulse import (
    ResolvedPulse,
    arm_sequencer,
    resolve_pulse,
)

from .calibration import CalibrationResult, FrameContract, TrapCalibration, calibrate


_INTEGRATION_METHODS = {"box", "psf", "uniform_psf"}
_THRESHOLD_METHODS = {"empirical", "gaussian"}
_REDUCERS = {"mean", "sum", "median", "max"}


def _positive_float(value: object, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _non_empty_key(value: object, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _roi(value: object | None) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    try:
        result = tuple(int(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError("roi_xywh must contain four integers or be None") from exc
    if len(result) != 4:
        raise ValueError("roi_xywh must contain four integers or be None")
    x, y, width, height = result
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("roi_xywh must have a non-negative origin and positive size")
    return x, y, width, height


@dataclass(frozen=True)
class CalibrationRequest:
    """One frozen calibration protocol and analysis request."""

    camera_key: str
    sequencer_key: str
    pulse_template: str
    repeats: int
    reference_exposure_seconds: float
    readout_exposure_seconds: float
    roi_xywh: tuple[int, int, int, int] | None
    integration_method: str
    threshold_method: str
    integration_half_width: int
    reducer: str
    detection_spot_sigma: float
    detection_min_distance: int
    detection_sigma: float
    timeout_seconds: float

    def __post_init__(self) -> None:
        camera_key = _non_empty_key(self.camera_key, "camera_key")
        sequencer_key = _non_empty_key(self.sequencer_key, "sequencer_key")
        pulse_template = _non_empty_key(self.pulse_template, "pulse_template")
        repeats = int(self.repeats)
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        reference_exposure = _positive_float(
            self.reference_exposure_seconds,
            "reference_exposure_seconds",
        )
        readout_exposure = _positive_float(
            self.readout_exposure_seconds,
            "readout_exposure_seconds",
        )
        if readout_exposure > reference_exposure:
            raise ValueError("readout exposure cannot exceed reference exposure")
        integration_method = str(self.integration_method).lower()
        if integration_method not in _INTEGRATION_METHODS:
            raise ValueError(
                "integration_method must be 'box', 'psf', or 'uniform_psf'"
            )
        threshold_method = str(self.threshold_method).lower()
        if threshold_method not in _THRESHOLD_METHODS:
            raise ValueError("threshold_method must be 'empirical' or 'gaussian'")
        integration_half_width = int(self.integration_half_width)
        if integration_half_width < 0:
            raise ValueError("integration_half_width must be non-negative")
        reducer = str(self.reducer).lower()
        if reducer not in _REDUCERS:
            raise ValueError("reducer must be mean, sum, median, or max")
        detection_spot_sigma = _positive_float(
            self.detection_spot_sigma,
            "detection_spot_sigma",
        )
        detection_min_distance = int(self.detection_min_distance)
        if detection_min_distance <= 0:
            raise ValueError("detection_min_distance must be positive")
        detection_sigma = _positive_float(self.detection_sigma, "detection_sigma")
        timeout_seconds = _positive_float(self.timeout_seconds, "timeout_seconds")
        object.__setattr__(self, "camera_key", camera_key)
        object.__setattr__(self, "sequencer_key", sequencer_key)
        object.__setattr__(self, "pulse_template", pulse_template)
        object.__setattr__(self, "repeats", repeats)
        object.__setattr__(self, "reference_exposure_seconds", reference_exposure)
        object.__setattr__(self, "readout_exposure_seconds", readout_exposure)
        object.__setattr__(self, "roi_xywh", _roi(self.roi_xywh))
        object.__setattr__(self, "integration_method", integration_method)
        object.__setattr__(self, "threshold_method", threshold_method)
        object.__setattr__(self, "integration_half_width", integration_half_width)
        object.__setattr__(self, "reducer", reducer)
        object.__setattr__(self, "detection_spot_sigma", detection_spot_sigma)
        object.__setattr__(self, "detection_min_distance", detection_min_distance)
        object.__setattr__(self, "detection_sigma", detection_sigma)
        object.__setattr__(self, "timeout_seconds", timeout_seconds)

    def to_dict(self) -> dict[str, object]:
        return {
            "camera_key": self.camera_key,
            "sequencer_key": self.sequencer_key,
            "pulse_template": self.pulse_template,
            "repeats": self.repeats,
            "reference_exposure_seconds": self.reference_exposure_seconds,
            "readout_exposure_seconds": self.readout_exposure_seconds,
            "roi_xywh": None if self.roi_xywh is None else list(self.roi_xywh),
            "integration_method": self.integration_method,
            "threshold_method": self.threshold_method,
            "integration_half_width": self.integration_half_width,
            "reducer": self.reducer,
            "detection_spot_sigma": self.detection_spot_sigma,
            "detection_min_distance": self.detection_min_distance,
            "detection_sigma": self.detection_sigma,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass(frozen=True)
class CalibrationCapture:
    """Adapter records from one exact long/readout/long acquisition."""

    cycles: tuple[
        tuple[CameraFrameRecord, CameraFrameRecord, CameraFrameRecord], ...
    ]
    terminal: CameraCaptureTerminalRecord

    def __post_init__(self) -> None:
        cycles = tuple(tuple(cycle) for cycle in self.cycles)
        if not cycles or any(
            len(cycle) != 3
            or any(not isinstance(frame, CameraFrameRecord) for frame in cycle)
            for cycle in cycles
        ):
            raise ValueError(
                "calibration capture requires non-empty three-frame cycles"
            )
        if not isinstance(self.terminal, CameraCaptureTerminalRecord):
            raise TypeError("terminal must be CameraCaptureTerminalRecord")
        object.__setattr__(self, "cycles", cycles)

    @property
    def frames(self) -> tuple[CameraFrameRecord, ...]:
        return tuple(frame for cycle in self.cycles for frame in cycle)

    @property
    def reference(self) -> tuple[tuple[CameraFrameRecord, CameraFrameRecord], ...]:
        return tuple((cycle[0], cycle[2]) for cycle in self.cycles)

    @property
    def short(self) -> tuple[CameraFrameRecord, ...]:
        return tuple(cycle[1] for cycle in self.cycles)


@dataclass(frozen=True)
class CalibrationRunResult:
    """Artifact and in-memory analysis returned by one task run."""

    artifact_path: Path
    calibration: TrapCalibration
    report: Mapping[str, Any]
    capture: CalibrationCapture
    reference: tuple[tuple[CameraFrameRecord, CameraFrameRecord], ...]
    short: tuple[CameraFrameRecord, ...]
    pulse: Mapping[str, object]
    run_record: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.calibration, TrapCalibration):
            raise TypeError("calibration must be TrapCalibration")
        if not isinstance(self.capture, CalibrationCapture):
            raise TypeError("capture must be CalibrationCapture")
        object.__setattr__(self, "artifact_path", Path(self.artifact_path).resolve())
        object.__setattr__(self, "report", dict(self.report))
        object.__setattr__(self, "reference", tuple(tuple(group) for group in self.reference))
        object.__setattr__(self, "short", tuple(self.short))
        object.__setattr__(self, "pulse", dict(self.pulse))
        object.__setattr__(self, "run_record", dict(self.run_record))


def _camera_snapshot(point: CameraWorkingPoint) -> dict[str, object]:
    roi_y, roi_x = point.roi_origin_yx
    roi_height, roi_width = point.roi_shape_yx
    return {
        "acquisition_mode": point.acquisition_mode,
        "frame_shape_yx": list(point.frame_shape_yx),
        "sensor_shape_yx": list(point.sensor_shape_yx),
        "roi_xywh": [roi_x, roi_y, roi_width, roi_height],
        "binning_yx": list(point.binning_yx),
        "dtype": point.dtype.str,
        "count_unit": point.count_unit,
        "exposure_seconds": point.exposure_seconds,
        "required_external_trigger_interval_seconds": (
            point.required_external_trigger_interval_seconds
        ),
        "external_trigger_integration_start_offset_seconds": (
            point.external_trigger_integration_start_offset_seconds
        ),
        "gain": point.gain,
        "readout_mode": point.readout_mode,
    }


def _plain(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    raise TypeError(f"device snapshot contains non-plain {type(value).__name__}")


def _sequencer_snapshot(sequencer: object) -> dict[str, object]:
    snapshot = sequencer.snapshot()
    if not isinstance(snapshot, Mapping):
        raise TypeError("sequencer snapshot must be a mapping")
    fields = (
        "opened",
        "loaded",
        "firing",
        "forever",
        "cursor",
        "scan_count",
        "underflow",
        "status",
    )
    return {
        key: _plain(snapshot[key])
        for key in fields
        if key in snapshot
    }


class CalibrationTask:
    """Drive one calibration protocol and return one saved artifact."""

    instance_id = "calibration"

    def __init__(
        self,
        *,
        camera: CameraAdapter,
        sequencer: object,
        request: CalibrationRequest,
        pulse_search_paths: Sequence[str | Path],
        artifact_directory: str | Path,
    ) -> None:
        if not isinstance(camera, CameraAdapter):
            raise TypeError("camera must implement CameraAdapter")
        for name in ("load", "fire", "wait_done", "safe", "snapshot"):
            if not callable(getattr(sequencer, name, None)):
                raise TypeError(f"sequencer must expose {name}")
        if not isinstance(request, CalibrationRequest):
            raise TypeError("request must be CalibrationRequest")
        paths = tuple(Path(value).expanduser().resolve() for value in pulse_search_paths)
        if not paths:
            raise ValueError("pulse_search_paths must not be empty")
        directory = Path(artifact_directory).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError("artifact_directory must be an existing directory")
        self.camera = camera
        self.sequencer = sequencer
        self._request = request
        self.pulse_search_paths = paths
        self.artifact_directory = directory
        self._actual_working_point: CameraWorkingPoint | None = None
        self._result: CalibrationRunResult | None = None

    @property
    def request(self) -> CalibrationRequest:
        return self._request

    @property
    def actual_working_point(self) -> CameraWorkingPoint | None:
        return self._actual_working_point

    @property
    def result(self) -> CalibrationRunResult | None:
        return self._result

    def _resolve_pulse(self) -> ResolvedPulse:
        return resolve_pulse(
            self.request.pulse_template,
            search_paths=self.pulse_search_paths,
            slot_values={
                "reference_before": self.request.reference_exposure_seconds,
                "readout": self.request.readout_exposure_seconds,
                "reference_after": self.request.reference_exposure_seconds,
            },
        )

    def _pulse_facts(self, pulse: ResolvedPulse) -> dict[str, object]:
        metadata = pulse.metadata
        if int(metadata.get("camera_windows", 0)) != 3:
            raise ValueError(
                f"calibration pulse {pulse.name!r} must declare exactly three camera windows"
            )
        if metadata.get("repeat_forever", False):
            raise ValueError(
                f"pulse {pulse.name!r} is repeat_forever and cannot finish calibration"
            )
        try:
            exposures = tuple(float(value) for value in metadata["frame_exposures"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "calibration pulse must declare frame_exposures=(long, readout, long)"
            ) from exc
        expected = (
            self.request.reference_exposure_seconds,
            self.request.readout_exposure_seconds,
            self.request.reference_exposure_seconds,
        )
        if (
            len(exposures) != 3
            or any(not np.isfinite(value) or value <= 0 for value in exposures)
            or not np.allclose(exposures, expected, rtol=1e-9, atol=1e-12)
        ):
            raise ValueError(
                "calibration pulse frame exposures do not match the frozen request"
            )
        semantics = tuple(metadata.get("frame_semantics", ()))
        if semantics != (
            "reference_long_before",
            "short_readout",
            "reference_long_after",
        ):
            raise ValueError("calibration pulse frame_semantics must be long/readout/long")
        if tuple(metadata.get("reference_frame_indices", ())) != (0, 2):
            raise ValueError("calibration pulse reference_frame_indices must be (0, 2)")
        if int(metadata.get("short_frame_index", -1)) != 1:
            raise ValueError("calibration pulse short_frame_index must be 1")
        return {
            "name": pulse.name,
            "path": None if pulse.path is None else str(pulse.path),
            "camera_trigger_channel": metadata.get("camera_trigger_channel"),
            "camera_windows": 3,
            "frame_exposures": list(exposures),
            "frame_semantics": list(semantics),
            "reference_frame_indices": [0, 2],
            "readout_frame_index": 1,
        }

    def _safe(self) -> None:
        self.sequencer.safe()

    def _capture(
        self,
        pulse: ResolvedPulse,
        *,
        context: object | None,
    ) -> tuple[CalibrationCapture, Mapping[str, object]]:
        count = self.request.repeats * 3
        armed = False
        try:
            self.camera.arm(
                count,
                source_group_sizes=(3,) * self.request.repeats,
                buffer_frame_count=count,
                timeout=self.request.timeout_seconds,
            )
            armed = True
            arm_sequencer(self.sequencer, pulse.program, pulse.metadata)
            sequencer_snapshot = _sequencer_snapshot(self.sequencer)
            cycles: list[
                tuple[CameraFrameRecord, CameraFrameRecord, CameraFrameRecord]
            ] = []
            for _ in range(self.request.repeats):
                if context is not None and context.cancel_requested():
                    raise RuntimeError("calibration was cancelled")
                self.sequencer.fire()
                report = self.sequencer.wait_done(self.request.timeout_seconds)
                if report is None:
                    raise TimeoutError(
                        "a calibration shot was fired and never reported done within "
                        f"{self.request.timeout_seconds:g}s"
                    )
                fault = str(getattr(report, "fault", ""))
                if fault:
                    raise RuntimeError(f"calibration shot: {fault}")
                records = tuple(
                    self.camera.read_frame_records(
                        3,
                        timeout=self.request.timeout_seconds,
                        exact=True,
                    )
                )
                if len(records) != 3 or any(
                    not isinstance(record, CameraFrameRecord) for record in records
                ):
                    raise RuntimeError(
                        "camera returned an incomplete calibration cycle"
                    )
                cycles.append((records[0], records[1], records[2]))
            terminal = self.camera.finish_record_capture()
            armed = False
            if (
                terminal.produced_count != count
                or not terminal.source_stopped
                or not terminal.no_more_frames
                or not terminal.joined
            ):
                raise RuntimeError("camera did not prove exact calibration completion")
            return CalibrationCapture(tuple(cycles), terminal), sequencer_snapshot
        except BaseException:
            if armed:
                self.camera.finish_record_capture()
            raise

    def _frame_contract(
        self,
        actual: CameraWorkingPoint,
        pulse: Mapping[str, object],
    ) -> FrameContract:
        roi_y, roi_x = actual.roi_origin_yx
        roi_height, roi_width = actual.roi_shape_yx
        frame_exposures = pulse["frame_exposures"]
        readout_gate = float(frame_exposures[1])  # type: ignore[index]
        return FrameContract(
            actual.frame_shape_yx,
            sensor_shape=actual.sensor_shape_yx,
            roi_xywh=(roi_x, roi_y, roi_width, roi_height),
            binning_yx=actual.binning_yx,
            exposure_seconds=min(actual.exposure_seconds, readout_gate),
            camera_id=self.request.camera_key,
            readout_mode=actual.readout_mode,
        )

    def _analyse(
        self,
        capture: CalibrationCapture,
        contract: FrameContract,
    ) -> CalibrationResult:
        return calibrate(
            capture.reference,
            capture.short,
            frame_contract=contract,
            method=self.request.integration_method,
            threshold_method=self.request.threshold_method,
            integration_half_width=self.request.integration_half_width,
            reducer=self.request.reducer,
            detection_spot_sigma=self.request.detection_spot_sigma,
            detection_min_distance=self.request.detection_min_distance,
            detection_sigma=self.request.detection_sigma,
        )

    def _run(self, context: object | None) -> CalibrationRunResult:
        self._actual_working_point = None
        self._result = None
        try:
            pulse = self._resolve_pulse()
            pulse_facts = self._pulse_facts(pulse)
            actual = self.camera.configure_measurement(
                exposure_seconds=self.request.reference_exposure_seconds,
                roi_xywh=self.request.roi_xywh,
            )
            if not isinstance(actual, CameraWorkingPoint):
                raise TypeError(
                    "camera configure_measurement must return CameraWorkingPoint"
                )
            self._actual_working_point = actual
            capture, sequencer_snapshot = self._capture(pulse, context=context)
            contract = self._frame_contract(actual, pulse_facts)
            analysis = self._analyse(capture, contract)
            run_record = {
                "request": self.request.to_dict(),
                "actual_devices": {
                    self.request.camera_key: _camera_snapshot(actual),
                    self.request.sequencer_key: dict(sequencer_snapshot),
                },
                "pulse": dict(pulse_facts),
            }
            artifact_report = dict(analysis.calibration.report)
            artifact_report["run_record"] = run_record
            calibration = TrapCalibration(
                analysis.calibration.site_map,
                analysis.calibration.readout_model,
                analysis.calibration.frame_contract,
                artifact_report,
            )
            artifact_path = unique_path(
                self.artifact_directory,
                "calibration",
                ".json",
            )
            calibration.save(artifact_path)
            result = CalibrationRunResult(
                artifact_path,
                calibration,
                analysis.report,
                capture,
                capture.reference,
                capture.short,
                pulse_facts,
                run_record,
            )
            self._result = result
            return result
        except BaseException:
            self._safe()
            raise

    def run(self) -> CalibrationRunResult:
        return self._run(None)

    def execute(self, context: object) -> CalibrationRunResult:
        return self._run(context)


__all__ = [
    "CalibrationCapture",
    "CalibrationRequest",
    "CalibrationRunResult",
    "CalibrationTask",
]
