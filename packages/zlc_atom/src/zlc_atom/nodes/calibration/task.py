"""Calibration orchestration over camera, sequencer, typed data, and artifact."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from zlc_data import AxisId, PointColumn, SITE, SPATIAL_X, SPATIAL_Y
from zlc_durable import unique_path
from zlc_pulse import PulseSequence

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

from .calibration import (
    CalibrationResult,
    FrameContract,
    ReadoutModelKind,
    TrapCalibration,
    calibrate,
)
from .outputs import (
    CAPTURE_PREVIEW_DECLARATION,
    CalibrationCapturePreviewSlot,
    _image_axis_specs,
    _snapshot,
)


_THRESHOLD_METHODS = {"empirical", "gaussian"}
_REDUCERS = {"mean", "sum", "median", "max"}
_DEVICE_WAIT_SECONDS = 2.0


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
    default_model_kind: ReadoutModelKind
    threshold_method: str
    box_half_width: int
    box_reducer: str
    psf_half_width: int
    psf_padding: int
    detection_spot_sigma: float
    detection_min_distance: int
    detection_sigma: float

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
        if not isinstance(self.default_model_kind, ReadoutModelKind):
            raise TypeError("default_model_kind must be ReadoutModelKind")
        threshold_method = str(self.threshold_method).lower()
        if threshold_method not in _THRESHOLD_METHODS:
            raise ValueError("threshold_method must be 'empirical' or 'gaussian'")
        box_half_width = int(self.box_half_width)
        psf_half_width = int(self.psf_half_width)
        if box_half_width < 0 or psf_half_width < 0:
            raise ValueError("integration half-widths must be non-negative")
        psf_padding = int(self.psf_padding)
        if psf_padding <= 0:
            raise ValueError("psf_padding must be positive")
        box_reducer = str(self.box_reducer).lower()
        if box_reducer not in _REDUCERS:
            raise ValueError("box_reducer must be mean, sum, median, or max")
        detection_spot_sigma = _positive_float(
            self.detection_spot_sigma,
            "detection_spot_sigma",
        )
        detection_min_distance = int(self.detection_min_distance)
        if detection_min_distance <= 0:
            raise ValueError("detection_min_distance must be positive")
        detection_sigma = _positive_float(self.detection_sigma, "detection_sigma")
        object.__setattr__(self, "camera_key", camera_key)
        object.__setattr__(self, "sequencer_key", sequencer_key)
        object.__setattr__(self, "pulse_template", pulse_template)
        object.__setattr__(self, "repeats", repeats)
        object.__setattr__(self, "reference_exposure_seconds", reference_exposure)
        object.__setattr__(self, "readout_exposure_seconds", readout_exposure)
        object.__setattr__(self, "roi_xywh", _roi(self.roi_xywh))
        object.__setattr__(self, "threshold_method", threshold_method)
        object.__setattr__(self, "box_half_width", box_half_width)
        object.__setattr__(self, "box_reducer", box_reducer)
        object.__setattr__(self, "psf_half_width", psf_half_width)
        object.__setattr__(self, "psf_padding", psf_padding)
        object.__setattr__(self, "detection_spot_sigma", detection_spot_sigma)
        object.__setattr__(self, "detection_min_distance", detection_min_distance)
        object.__setattr__(self, "detection_sigma", detection_sigma)

    def to_dict(self) -> dict[str, object]:
        return {
            "camera_key": self.camera_key,
            "sequencer_key": self.sequencer_key,
            "pulse_template": self.pulse_template,
            "repeats": self.repeats,
            "reference_exposure_seconds": self.reference_exposure_seconds,
            "readout_exposure_seconds": self.readout_exposure_seconds,
            "roi_xywh": None if self.roi_xywh is None else list(self.roi_xywh),
            "default_model_kind": self.default_model_kind.value,
            "threshold_method": self.threshold_method,
            "box_half_width": self.box_half_width,
            "box_reducer": self.box_reducer,
            "psf_half_width": self.psf_half_width,
            "psf_padding": self.psf_padding,
            "detection_spot_sigma": self.detection_spot_sigma,
            "detection_min_distance": self.detection_min_distance,
            "detection_sigma": self.detection_sigma,
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


def _save_report_images(result: CalibrationRunResult) -> Path:
    """Render the SiteMap and each readout model directly through zlc_plot."""

    from zlc_plot import (
        AxisRef,
        HistogramPlot,
        ImagePointOverlay,
        PlotLabels,
        PointStatus,
        facet_grid,
        image,
    )

    calibration = result.calibration
    site_map = calibration.site_map
    report_root = result.artifact_path.with_suffix("") / "report"
    report_root.mkdir(parents=True)
    generation = result.artifact_path.stem
    revision = len(result.capture.cycles)

    site_map_snapshot = _snapshot(
        np.asarray(result.report["reference_average"], dtype="<f8")[np.newaxis, ...],
        signal="site_map",
        roles=(SPATIAL_Y, SPATIAL_X),
        axis_specs=_image_axis_specs(
            calibration.frame_contract.image_shape,
            site_map.coordinate_frame,
        ),
        generation=generation,
        revision=revision,
    )
    overlay = ImagePointOverlay(
        revision,
        site_map.centers_xy,
        point_ids=site_map.site_ids,
        labels=tuple(str(index + 1) for index in range(site_map.n_sites)),
        statuses=tuple(
            PointStatus.UNKNOWN if valid else PointStatus.INVALID
            for valid in site_map.valid_sites
        ),
    )
    with image(
        site_map_snapshot,
        AxisRef.data("calibration.image.x"),
        AxisRef.data("calibration.image.y"),
        overlay=overlay,
        labels=PlotLabels(title="Site map", x="x (pixel)", y="y (pixel)"),
        size="4x4",
    ) as plot:
        plot.save(report_root / "site_map.png")

    site_column = PointColumn(
        AxisId("calibration.site"),
        "site",
        SITE,
        PointColumn.TEXT,
        site_map.site_ids,
        coordinate_labels=tuple(
            str(index) for index in range(1, site_map.n_sites + 1)
        ),
    )
    labels_valid = np.asarray(result.report["labels_valid"], dtype=bool)
    model_reports = result.report["models"]
    for model in calibration.models:
        model_report = model_reports[model.kind.value]
        short_signals = np.asarray(model_report["short_signals"], dtype="<f8")
        site_valid = (
            site_map.valid_sites
            & model.usable_sites
            & np.isfinite(model.thresholds)
        )
        samples = _snapshot(
            short_signals,
            signal=f"{model.kind.value}_readout_samples",
            roles=(SITE,),
            point_columns={SITE: site_column},
            generation=generation,
            revision=revision,
            cell_validity=(
                labels_valid
                & np.isfinite(short_signals)
                & site_valid[np.newaxis, :]
            ),
        )
        thresholds = tuple(
            float(value) if valid else None
            for value, valid in zip(model.thresholds, site_valid, strict=True)
        )
        title = f"{model.kind.value.replace('_', ' ')} readout"
        with facet_grid(
            samples,
            AxisRef.point("calibration.site"),
            HistogramPlot(PlotLabels(x="Readout signal", y="Count")),
            labels=PlotLabels(title=title),
            size="4x4",
        ) as plot:
            plot.set_facet_thresholds(thresholds, display=False)
            plot.fit("bimodal_gaussian", live=False, fit_all_facets=True)
            plot.save(report_root / f"{model.kind.value}.png")

    return report_root


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
    """Drive one protocol, publish its preview, and save its result and plots."""

    instance_id = "calibration"

    def __init__(
        self,
        *,
        camera: CameraAdapter,
        sequencer: object,
        request: CalibrationRequest,
        pulse_sequence: PulseSequence,
        pulse_path: str | Path,
        artifact_directory: str | Path,
    ) -> None:
        if not isinstance(camera, CameraAdapter):
            raise TypeError("camera must implement CameraAdapter")
        for name in ("describe", "load", "fire", "wait_done", "safe", "snapshot"):
            if not callable(getattr(sequencer, name, None)):
                raise TypeError(f"sequencer must expose {name}")
        if not isinstance(request, CalibrationRequest):
            raise TypeError("request must be CalibrationRequest")
        if not isinstance(pulse_sequence, PulseSequence):
            raise TypeError("pulse_sequence must be PulseSequence")
        directory = Path(artifact_directory).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError("artifact_directory must be an existing directory")
        self.camera = camera
        self.sequencer = sequencer
        self._request = request
        self.pulse_sequence = pulse_sequence
        self.pulse_path = Path(pulse_path).expanduser().resolve()
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

    @property
    def dataset_output_declarations(self):
        return (CAPTURE_PREVIEW_DECLARATION,)

    def _resolve_pulse(self) -> ResolvedPulse:
        return resolve_pulse(
            self.pulse_sequence,
            path=self.pulse_path,
            board=self.sequencer.describe(),
            api_values={
                "reference_probe_duration_before": self.request.reference_exposure_seconds,
                "readout_probe_duration": self.request.readout_exposure_seconds,
                "reference_probe_duration_after": self.request.reference_exposure_seconds,
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
        actual: CameraWorkingPoint,
        pulse_facts: Mapping[str, object],
    ) -> tuple[
        CalibrationCapture,
        Mapping[str, object],
    ]:
        count = self.request.repeats * 3
        armed = False
        try:
            self.camera.arm(
                count,
                source_group_sizes=(3,) * self.request.repeats,
                buffer_frame_count=count,
                timeout=_DEVICE_WAIT_SECONDS,
            )
            armed = True
            arm_sequencer(self.sequencer, pulse)
            sequencer_snapshot = _sequencer_snapshot(self.sequencer)
            run_record = self._run_record(
                actual,
                sequencer_snapshot,
                pulse_facts,
            )
            preview: CalibrationCapturePreviewSlot | None = None
            if context is not None:
                preview = CalibrationCapturePreviewSlot(
                    repeats=self.request.repeats,
                    frame_shape=actual.frame_shape_yx,
                    dtype=actual.dtype,
                    generation=context.generation,
                    run_record=run_record,
                )
                context.attach_live_outputs(preview)
                context.report_progress(
                    "Capturing calibration",
                    current=0,
                    total=self.request.repeats,
                )
            cycles: list[
                tuple[CameraFrameRecord, CameraFrameRecord, CameraFrameRecord]
            ] = []
            for _ in range(self.request.repeats):
                if context is not None and context.cancel_requested():
                    raise RuntimeError("calibration was cancelled")
                self.sequencer.fire()
                report = self.sequencer.wait_done(_DEVICE_WAIT_SECONDS)
                if report is None:
                    raise TimeoutError(
                        "a calibration shot was fired and never reported done within "
                        f"{_DEVICE_WAIT_SECONDS:g}s"
                    )
                fault = str(getattr(report, "fault", ""))
                if fault:
                    raise RuntimeError(f"calibration shot: {fault}")
                records = tuple(
                    self.camera.read_frame_records(
                        3,
                        timeout=_DEVICE_WAIT_SECONDS,
                        exact=True,
                    )
                )
                if len(records) != 3 or any(
                    not isinstance(record, CameraFrameRecord) for record in records
                ):
                    raise RuntimeError(
                        "camera returned an incomplete calibration cycle"
                    )
                cycle = (records[0], records[1], records[2])
                cycles.append(cycle)
                if preview is not None:
                    preview.update(cycle)
                    context.report_progress(
                        "Capturing calibration",
                        current=len(cycles),
                        total=self.request.repeats,
                    )
            terminal = self.camera.finish_record_capture()
            armed = False
            if (
                terminal.produced_count != count
                or not terminal.source_stopped
                or not terminal.no_more_frames
                or not terminal.joined
            ):
                raise RuntimeError("camera did not prove exact calibration completion")
            return (
                CalibrationCapture(tuple(cycles), terminal),
                run_record,
            )
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

    def _run_record(
        self,
        actual: CameraWorkingPoint,
        sequencer_snapshot: Mapping[str, object],
        pulse_facts: Mapping[str, object],
    ) -> dict[str, object]:
        return {
            "request": self.request.to_dict(),
            "actual_devices": {
                self.request.camera_key: _camera_snapshot(actual),
                self.request.sequencer_key: dict(sequencer_snapshot),
            },
            "pulse": dict(pulse_facts),
        }

    def _analyse(
        self,
        capture: CalibrationCapture,
        contract: FrameContract,
    ) -> CalibrationResult:
        return calibrate(
            capture.reference,
            capture.short,
            frame_contract=contract,
            default_model_kind=self.request.default_model_kind,
            threshold_method=self.request.threshold_method,
            box_half_width=self.request.box_half_width,
            box_reducer=self.request.box_reducer,
            psf_half_width=self.request.psf_half_width,
            psf_padding=self.request.psf_padding,
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
            capture, run_record = self._capture(
                pulse,
                context=context,
                actual=actual,
                pulse_facts=pulse_facts,
            )
            contract = self._frame_contract(actual, pulse_facts)
            if context is not None:
                context.report_progress("Analysing calibration")
            analysis = self._analyse(capture, contract)
            artifact_report = dict(analysis.calibration.report)
            artifact_report["run_record"] = run_record
            calibration = TrapCalibration(
                analysis.calibration.site_map,
                analysis.calibration.models,
                analysis.calibration.default_model_kind,
                analysis.calibration.frame_contract,
                artifact_report,
            )
            artifact_path = unique_path(
                self.artifact_directory,
                "calibration",
                ".json",
            )
            if context is not None:
                context.report_progress("Saving calibration")
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
            if context is not None:
                context.report_progress("Saving calibration report")
            _save_report_images(result)
            self._result = result
            if context is not None:
                context.report_progress(
                    "Calibration complete",
                    current=self.request.repeats,
                    total=self.request.repeats,
                )
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
