"""Calibration orchestration over camera, sequencer, typed data, and artifact."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from zlc_data import (
    COMPONENT,
    SITE,
    SPATIAL_X,
    SPATIAL_Y,
    AxisId,
    AxisSpec,
)
from zlc_durable import unique_path
from zlc_pulse import PulseSequence, convert_time

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


def _positive_float(value: object, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _slot_number(value: object, name: str) -> int:
    """One 1-based API slot number, as an operator counts them."""

    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive slot number")
    return result


def _non_empty_key(value: object, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


@dataclass(frozen=True)
class CalibrationRequest:
    """One frozen calibration protocol and analysis request."""

    camera_key: str
    sequencer_key: str
    pulse_template: str
    repeats: int
    reference_exposure_seconds: float
    readout_exposure_seconds: float
    #: Which of the pulse's API slots this protocol drives, by number, and the
    #: port the board gates its camera from.  By NUMBER because a slot's name
    #: belongs to whoever wrote the pulse: a node that matched names could only
    #: ever run the one template it shipped with, however many slots an
    #: operator's own imaging pulse offered.
    reference_before_slot: int
    readout_slot: int
    reference_after_slot: int
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
        if readout_exposure >= reference_exposure:
            # Not a rule about numbers: the readout frame is recognised in the
            # compiled program by being the SHORT one, so a run whose three
            # windows are the same length has no readout frame to find.
            raise ValueError("readout exposure must be shorter than the reference exposure")
        slots = tuple(
            _slot_number(value, name)
            for value, name in (
                (self.reference_before_slot, "reference_before_slot"),
                (self.readout_slot, "readout_slot"),
                (self.reference_after_slot, "reference_after_slot"),
            )
        )
        if len(set(slots)) != len(slots):
            raise ValueError(
                "the three calibration exposures must be driven by three "
                "different pulse API slots"
            )
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
        object.__setattr__(self, "reference_before_slot", slots[0])
        object.__setattr__(self, "readout_slot", slots[1])
        object.__setattr__(self, "reference_after_slot", slots[2])
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
            "reference_before_slot": self.reference_before_slot,
            "readout_slot": self.readout_slot,
            "reference_after_slot": self.reference_after_slot,
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


#: The acquisition order this protocol runs in: a long reference frame, the
#: short readout between them, a long reference after.  THE statement of it --
#: it used to be repeated in the resolver's metadata, re-checked in the facts,
#: written back as a literal, and indexed again in two more places.
REFERENCE_FRAME_INDICES = (0, 2)
READOUT_FRAME_INDEX = 1


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
        first, second = REFERENCE_FRAME_INDICES
        return tuple((cycle[first], cycle[second]) for cycle in self.cycles)

    @property
    def short(self) -> tuple[CameraFrameRecord, ...]:
        return tuple(cycle[READOUT_FRAME_INDEX] for cycle in self.cycles)


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
        ImagePlot,
        ImagePointOverlay,
        PlotLabels,
        PointStatus,
        curve,
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
        statuses={
            None: tuple(
                PointStatus.UNKNOWN if valid else PointStatus.INVALID
                for valid in site_map.valid_sites
            )
        },
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

    # ONE declaration of site identity, owned by the SiteMap that measured it.
    # A site array is one image resampled onto the trap lattice: cell data, not
    # a scan.  This replaces both a TEXT site point column (unplottable) and the
    # numeric "site ordinal" point column invented to work around it.
    site_axis = site_map.site_axis
    site_axis_id = site_axis.axis_id
    labels_valid = np.asarray(result.report["labels_valid"], dtype=bool)
    model_reports = result.report["models"]

    model_names = tuple(model.kind.value for model in calibration.models)
    model_axis = AxisSpec(
        AxisId("calibration.model"),
        "readout model",
        COMPONENT,
        len(model_names),
        coordinate_labels=tuple(name.replace("_", " ") for name in model_names),
    )
    fidelity_values = np.stack(
        [
            np.asarray(model_reports[name]["site_fidelity"], dtype="<f8")
            for name in model_names
        ],
        axis=-1,
    )[np.newaxis, ...]
    fidelity_valid = np.stack(
        [
            site_map.valid_sites
            & model.usable_sites
            & np.isfinite(fidelity_values[0, :, index])
            for index, model in enumerate(calibration.models)
        ],
        axis=-1,
    )[np.newaxis, ...]
    # Nothing was scanned to make this: one number per (site, model) out of one
    # capture.  Both are cell axes, the point axis is empty, and per-site
    # usability now rides the axis it actually varies over.
    fidelity_snapshot = _snapshot(
        fidelity_values,
        signal="readout_fidelity",
        roles=(SITE, COMPONENT),
        axis_specs={SITE: site_axis, COMPONENT: model_axis},
        generation=generation,
        revision=revision,
        validity_axis_ids=(site_axis_id, model_axis.axis_id),
        validity_mask=fidelity_valid[:, np.newaxis, ...],
    )
    with curve(
        fidelity_snapshot,
        AxisRef.data("calibration.site"),
        group=AxisRef.data("calibration.model"),
        labels=PlotLabels(
            title="Held-out fidelity by readout model",
            x="Site",
            y="Fidelity",
        ),
        size="4x4",
    ) as plot:
        plot.save(report_root / "fidelity.png")

    for model in calibration.models:
        model_report = model_reports[model.kind.value]
        short_signals = np.asarray(model_report["short_signals"], dtype="<f8")
        site_valid = (
            site_map.valid_sites
            & model.usable_sites
            & np.isfinite(model.thresholds)
        )
        # The repeats ARE the repeat axis -- same conditions, measured again --
        # and the sites are the cell.  No independent variable was swept, so
        # there is no point axis to invent one on.
        samples = _snapshot(
            short_signals,
            signal=f"{model.kind.value}_readout_samples",
            roles=(SITE,),
            axis_specs={SITE: site_axis},
            generation=generation,
            revision=revision,
            validity_axis_ids=(site_axis_id,),
            validity_mask=(
                labels_valid
                & np.isfinite(short_signals)
                & site_valid[np.newaxis, :]
            )[:, np.newaxis, :],
        )
        thresholds = tuple(
            float(value) if valid else None
            for value, valid in zip(model.thresholds, site_valid, strict=True)
        )
        title = f"{model.kind.value.replace('_', ' ')} readout"
        with facet_grid(
            samples,
            AxisRef.data("calibration.site"),
            HistogramPlot(PlotLabels(x="Readout signal", y="Count")),
            labels=PlotLabels(title=title),
            size="4x4",
        ) as plot:
            plot.configure(
                parameters={"threshold_classifier": True},
                classifier_thresholds=thresholds,
            )
            plot.save(report_root / f"{model.kind.value}.png")

    psf_model = calibration.select_model(ReadoutModelKind.PER_SITE_PSF)
    psf_kernels = np.asarray(psf_model.psf_weights, dtype="<f8")
    kernel_height, kernel_width = psf_kernels.shape[-2:]
    kernel_valid = (
        site_map.valid_sites
        & psf_model.usable_sites
        & np.all(np.isfinite(psf_kernels), axis=(-2, -1))
    )[np.newaxis, :]
    psf_snapshot = _snapshot(
        psf_kernels[np.newaxis, ...],
        signal="psf_kernels",
        roles=(SITE, SPATIAL_Y, SPATIAL_X),
        axis_specs={
            SITE: site_axis,
            SPATIAL_Y: AxisSpec(
                AxisId("calibration.psf.y"),
                "y",
                SPATIAL_Y,
                kernel_height,
                coordinates=tuple(
                    range(-(kernel_height // 2), -(kernel_height // 2) + kernel_height)
                ),
                unit="pixel",
            ),
            SPATIAL_X: AxisSpec(
                AxisId("calibration.psf.x"),
                "x",
                SPATIAL_X,
                kernel_width,
                coordinates=tuple(
                    range(-(kernel_width // 2), -(kernel_width // 2) + kernel_width)
                ),
                unit="pixel",
            ),
        },
        generation=generation,
        revision=revision,
        validity_axis_ids=(site_axis_id,),
        validity_mask=kernel_valid[:, np.newaxis, :],
    )
    with facet_grid(
        psf_snapshot,
        AxisRef.data("calibration.site"),
        ImagePlot(
            AxisRef.data("calibration.psf.x"),
            AxisRef.data("calibration.psf.y"),
            labels=PlotLabels(x="x (pixel)", y="y (pixel)", value="Weight"),
        ),
        labels=PlotLabels(title="Per-site PSF kernels"),
        size="4x4",
    ) as plot:
        plot.save(report_root / "psf_kernels.png")

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
            api_values=self._driven_values(),
        )

    def _driven_values(self) -> dict[str, float]:
        """The three exposures, each said in the unit its slot declares.

        A parameter is written in ITS OWN unit.  This task holds SI seconds --
        the camera speaks nothing else -- so a slot declared in microseconds
        used to receive a number a million times too small, silently, and the
        shipped template only escaped it by declaring seconds itself.
        """

        values: dict[str, float] = {}
        for parameter, seconds in zip(
            self._driven_slots(),
            (
                self.request.reference_exposure_seconds,
                self.request.readout_exposure_seconds,
                self.request.reference_exposure_seconds,
            ),
            strict=True,
        ):
            if parameter.unit == "value":
                raise ValueError(
                    f"API slot {parameter.parameter_id!r} sets a DAC code, not "
                    "a duration; calibration drives the three probe lengths"
                )
            values[parameter.parameter_id] = convert_time(
                seconds, "s", parameter.unit
            )
        return values

    def _driven_slots(self) -> tuple[object, object, object]:
        """The three API slots this run drives, as the pulse declares them.

        Addressed by number, in the acquisition's own order: slot k is the
        k-th parameter the pulse declares, whatever its author called it and
        whichever unit they wrote it in.
        """

        declared = tuple(self.pulse_sequence.api_parameters)
        wanted = (
            self.request.reference_before_slot,
            self.request.readout_slot,
            self.request.reference_after_slot,
        )
        if max(wanted) > len(declared):
            raise ValueError(
                f"pulse {self.pulse_sequence.name!r} offers {len(declared)} "
                f"API slot(s) and this calibration drives slot {max(wanted)}"
            )
        return tuple(declared[slot - 1] for slot in wanted)

    def _pulse_facts(self, pulse: ResolvedPulse) -> dict[str, object]:
        """What this run will be remembered by: the pulse it played, and where.

        It does NOT re-read the compiled program to decide what the frames
        mean.  The three exposures went into three API slots the operator
        chose by number, and the acquisition reads them back in that same
        order; a task that re-derived the meaning from window counts and
        exposure lengths made itself depend on the shape of a document it does
        not own, and broke the moment the pulse was somebody else's.
        """

        return {
            "name": pulse.name,
            "path": None if pulse.path is None else str(pulse.path),
            "api_slots": [
                self.request.reference_before_slot,
                self.request.readout_slot,
                self.request.reference_after_slot,
            ],
            "api_parameters": [
                parameter.parameter_id for parameter in self._driven_slots()
            ],
            # What was actually written to the board, in the units it was
            # written in: an archive that only records seconds cannot say
            # whether the run agreed with the pulse.
            "api_values": self._driven_values(),
            "api_units": [
                parameter.unit for parameter in self._driven_slots()
            ],
            "frame_exposures": [
                self.request.reference_exposure_seconds,
                self.request.readout_exposure_seconds,
                self.request.reference_exposure_seconds,
            ],
            "reference_frame_indices": list(REFERENCE_FRAME_INDICES),
            "readout_frame_index": READOUT_FRAME_INDEX,
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
        firing = False
        try:
            self.camera.arm(
                count,
                source_group_sizes=(3,) * self.request.repeats,
                buffer_frame_count=count,
                timeout=self.camera.timeout,
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
            # One pulse for the whole run.  The board repeats the cycle on
            # its own and the camera is armed for every frame of it, so the
            # host has nothing to synchronise: it reads cycles until it has
            # them all and then takes the pulse down.  Firing each shot and
            # waiting for DONE made every repeat cost a round trip to the
            # board and a handshake, serialised the sequence against the
            # camera's own transfer, and could fail a run whose frames were
            # perfectly fine because a report arrived late.
            self.sequencer.fire(forever=True)
            firing = True
            for _ in range(self.request.repeats):
                if context is not None and context.cancel_requested():
                    raise RuntimeError("calibration was cancelled")
                records = tuple(
                    self.camera.read_frame_records(
                        3,
                        timeout=self.camera.timeout,
                        exact=True,
                    )
                )
                if len(records) != 3 or any(
                    not isinstance(record, CameraFrameRecord) for record in records
                ):
                    # The sensor's own answer to "how often can you be
                    # triggered here" is the number that decides this, so it
                    # is the number the operator is given.
                    interval = actual.required_external_trigger_interval_seconds
                    raise RuntimeError(
                        f"the camera returned {len(records)} frame(s) of a "
                        f"three-frame cycle: at this working point it "
                        f"integrates {actual.exposure_seconds:g}s per trigger "
                        f"and accepts one only every "
                        f"{'unknown' if interval is None else format(interval, 'g') + 's'}"
                        ", and a trigger arriving before that is ignored -- "
                        "the pulse must space its camera windows by more than "
                        "that, or the exposure must come down"
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
            self.sequencer.safe()
            firing = False
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
            if firing:
                self._safe()
            raise

    def _frame_contract(
        self,
        actual: CameraWorkingPoint,
        pulse: Mapping[str, object],
    ) -> FrameContract:
        roi_y, roi_x = actual.roi_origin_yx
        roi_height, roi_width = actual.roi_shape_yx
        # The gate is what this run ASKED for, from the request that froze it.
        # Reading it back out of the pulse only asked the same question of a
        # less reliable witness.
        readout_gate = self.request.readout_exposure_seconds
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
            # The exposure is this run's to state: the three frames are cut
            # by the pulse THIS task plays, and a camera still integrating
            # when the next trigger arrives simply ignores it -- which is one
            # frame short of a cycle, every cycle.  The geometry is the
            # bench's and is not touched: an ROI tuned around the traps has to
            # survive a calibration, and what was actually used is recorded in
            # the run record either way.
            actual = self.camera.set_exposure_seconds(
                self.request.reference_exposure_seconds
            )
            if not isinstance(actual, CameraWorkingPoint):
                raise TypeError(
                    "camera set_exposure_seconds must return CameraWorkingPoint"
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
