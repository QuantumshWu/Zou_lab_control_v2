"""Calibration orchestration over camera, sequencer, typed data, and artifact."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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
from zlc_data.figure_archive import figure_bytes, read_archive, read_dataset
from zlc_durable import (
    atomic_write_bytes,
    durable_makedirs,
    unique_path,
    write_readable_json,
)
from zlc_pulse import PulseSequence, convert_time

from zlc_atom.devices.camera.contract import (
    CameraAdapter,
    CameraCaptureTerminalRecord,
    CameraFrameRecord,
    CameraWorkingPoint,
)
from zlc_atom.devices.camera.photoelectrons import PHOTOELECTRONS
from zlc_atom.nodes.camera_measurement.measurement import (
    CameraMeasurementNode,
    CameraMeasurementRequest,
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
from .summary import readout_summary, summary_lines
from .outputs import (
    CAPTURE_PREVIEW_DECLARATION,
    capture_preview_output,
    cycle_snapshot,
    _image_axis_specs,
    _snapshot,
)


#: Where a calibration's frames come from.  Taking them and reading them back
#: are the same protocol either way: three frames a sample, the long ones
#: either side of the short one.  What differs is only whether the camera is
#: asked -- so a folder of saved samples can be calibrated again, with other
#: detection settings, without holding the bench.
FRAMES_FROM_CAMERA = "new"
FRAMES_FROM_FOLDER = "saved"
_FRAME_SOURCES = frozenset({FRAMES_FROM_CAMERA, FRAMES_FROM_FOLDER})

#: What one saved sample is called: one typed figure archive plus its rendered
#: picture, in the shared format the figure viewer opens.
SAVED_SAMPLE_STEM = "sample"

_THRESHOLD_METHODS = {"empirical", "gaussian"}
_REDUCERS = {"mean", "sum", "median", "max"}


def _roi_xywh(point: CameraWorkingPoint) -> tuple[int, int, int, int]:
    """The working point's rectangle, in the order an operator writes it."""

    roi_y, roi_x = point.roi_origin_yx
    roi_height, roi_width = point.roi_shape_yx
    return int(roi_x), int(roi_y), int(roi_width), int(roi_height)


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
    detection_sigma: float
    #: Read the camera in photoelectrons instead of counts.  Calibration
    #: keeps no conversion of its own -- it asks the measurement for the
    #: frames it wants -- and the thresholds it fits are in whatever it got.
    photoelectrons: bool = False
    #: Whether every acquired sample is written to disk as it arrives.
    save_frames: bool = False
    #: Where the frames come from: the camera, or a folder of saved samples.
    frame_source: str = FRAMES_FROM_CAMERA
    #: That folder, when they come from one.
    saved_frames_path: str = ""

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
        detection_sigma = _positive_float(self.detection_sigma, "detection_sigma")
        frame_source = str(self.frame_source).strip().lower()
        if frame_source not in _FRAME_SOURCES:
            raise ValueError(
                "frame_source must be "
                + " or ".join(repr(value) for value in sorted(_FRAME_SOURCES))
            )
        saved_frames_path = str(self.saved_frames_path).strip()
        if frame_source == FRAMES_FROM_FOLDER and not saved_frames_path:
            raise ValueError(
                "calibrating from saved frames needs the folder they were saved in"
            )
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
        object.__setattr__(self, "detection_sigma", detection_sigma)
        object.__setattr__(self, "photoelectrons", bool(self.photoelectrons))
        object.__setattr__(self, "save_frames", bool(self.save_frames))
        object.__setattr__(self, "frame_source", frame_source)
        object.__setattr__(self, "saved_frames_path", saved_frames_path)

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
            "detection_sigma": self.detection_sigma,
            PHOTOELECTRONS: self.photoelectrons,
            "save_frames": self.save_frames,
            "frame_source": self.frame_source,
            "saved_frames_path": self.saved_frames_path,
        }


#: The acquisition order this protocol runs in: a long reference frame, the
#: short readout between them, a long reference after.  THE statement of it --
#: it used to be repeated in the resolver's metadata, re-checked in the facts,
#: written back as a literal, and indexed again in two more places.
REFERENCE_FRAME_INDICES = (0, 2)
READOUT_FRAME_INDEX = 1


class SampleWriter:
    """Every acquired sample on disk, written as it arrives.

    Not at the end.  A run that is cancelled, that loses the bench, or that
    fails its analysis has still cost the atoms it cost, and the frames are
    the only part of it that cannot be recomputed -- so sample k is on disk
    before sample k+1 is asked for.

    The DATA is written in the loop; the pictures are rendered after it.  A
    matplotlib figure costs a few hundred milliseconds, which inside the read
    loop is long enough for the camera's ring to overrun and lose a frame,
    and a picture is derived from numbers that are already safe.
    """

    def __init__(
        self,
        folder: Path,
        *,
        working_point: CameraWorkingPoint,
        run_record: Mapping[str, object],
        generation: object,
    ) -> None:
        self.folder = Path(folder)
        durable_makedirs(self.folder)
        self._point = working_point
        self._run_record = dict(run_record)
        self._generation = generation
        self._samples: dict[int, Path] = {}

    def _paths(self, index: int) -> tuple[Path, Path]:
        name = f"{SAVED_SAMPLE_STEM}_{int(index):04d}"
        image_path = self.folder / f"{name}.png"
        return image_path, image_path.with_suffix(".npz")

    def write(self, index: int, cycle: Sequence[CameraFrameRecord]) -> Path:
        """Write one sample's numbers, and hold it for its picture afterwards.

        The shared figure encoder stores the typed dataset and its run record,
        so the figure viewer can reopen the numbers with their exact axes and
        the replay can recover the exposure, ROI and binning without importing
        Workbench panel state into this science plugin.
        """

        snapshot = cycle_snapshot(
            cycle,
            frame_shape=self._point.frame_shape_yx,
            origin_yx=self._point.roi_origin_yx,
            binning_yx=self._point.binning_yx,
            generation=self._generation,
            revision=int(index) + 1,
        )
        image_path, archive_path = self._paths(index)
        atomic_write_bytes(
            archive_path,
            figure_bytes(
                image_path.name,
                arrays={"data": snapshot},
                sections={"run_chain": [dict(self._run_record)]},
            ),
        )
        self._samples[int(index)] = archive_path
        return archive_path

    def render(self) -> int:
        """Draw every written sample, once the camera is no longer waiting."""

        from zlc_plot import AxisRef, ImagePlot, PlotLabels, facet_grid

        for index, archive_path in sorted(self._samples.items()):
            info, arrays = read_archive(archive_path)
            snapshot = read_dataset(info, arrays, "data")
            image_path, _archive_path = self._paths(index)
            with facet_grid(
                snapshot,
                AxisRef.point("calibration.capture_preview.frame"),
                ImagePlot(
                    AxisRef.data("calibration.image.x"),
                    AxisRef.data("calibration.image.y"),
                ),
                labels=PlotLabels(
                    title=f"calibration {SAVED_SAMPLE_STEM}_{int(index):04d}"
                ),
                size="4x4",
            ) as plot:
                plot.save(image_path)
        return len(self._samples)


def read_saved_samples(
    folder: str | Path,
) -> tuple[
    tuple[tuple[CameraFrameRecord, CameraFrameRecord, CameraFrameRecord], ...],
    dict[str, object],
]:
    """Every sample saved in a folder, in the order they were taken.

    Returns the cycles and the run record they were acquired under -- the same
    record the run wrote, which is where the exposure and the crop are, so a
    replay is calibrated against the geometry it was actually taken with.
    """

    directory = Path(folder).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"saved frames folder does not exist: {directory}")
    indexed_paths: list[tuple[int, Path]] = []
    for path in directory.glob(f"{SAVED_SAMPLE_STEM}_*.npz"):
        index_text = path.stem.removeprefix(f"{SAVED_SAMPLE_STEM}_")
        if not index_text.isdigit():
            raise ValueError(f"saved sample has a non-numeric index: {path.name}")
        index = int(index_text)
        if path.name != f"{SAVED_SAMPLE_STEM}_{index:04d}.npz":
            raise ValueError(f"saved sample has a non-canonical name: {path.name}")
        indexed_paths.append((index, path))
    indexed_paths.sort()
    if not indexed_paths:
        raise ValueError(
            f"{directory} holds no saved calibration samples "
            f"({SAVED_SAMPLE_STEM}_*.npz)"
        )
    cycles: list[
        tuple[CameraFrameRecord, CameraFrameRecord, CameraFrameRecord]
    ] = []
    run_record: dict[str, object] | None = None
    first_schema = None
    first_block_id = None
    first_generation = None
    for ordinal, (saved_index, path) in enumerate(indexed_paths):
        if saved_index != ordinal:
            raise ValueError(
                "saved calibration sample indices must be contiguous from zero; "
                f"expected {ordinal:04d}, got {saved_index:04d}"
            )
        info, arrays = read_archive(path)
        snapshot = read_dataset(info, arrays, "data")
        if info["name"] != f"{SAVED_SAMPLE_STEM}_{ordinal:04d}.png":
            raise ValueError(f"{path.name} metadata names another sample")
        if snapshot.ref.revision.value != ordinal + 1:
            raise ValueError(f"{path.name} carries the wrong dataset revision")
        if first_schema is None:
            first_schema = snapshot.block.schema
            first_block_id = snapshot.ref.block_id
            first_generation = snapshot.ref.stream_generation
        elif (
            snapshot.block.schema != first_schema
            or snapshot.ref.block_id != first_block_id
            or snapshot.ref.stream_generation != first_generation
        ):
            raise ValueError(f"{path.name} belongs to a different saved capture")
        values = np.asarray(snapshot.block.values)
        if values.ndim != 4 or values.shape[0] != 1 or values.shape[1] != 3:
            raise ValueError(
                f"{path.name} does not hold one three-frame sample: "
                f"its data is {values.shape}"
            )
        cycles.append(
            (
                CameraFrameRecord(values[0, 0], ordinal * 3),
                CameraFrameRecord(values[0, 1], ordinal * 3 + 1),
                CameraFrameRecord(values[0, 2], ordinal * 3 + 2),
            )
        )
        chain = info["sections"].get("run_chain")
        if (
            type(chain) is not list
            or len(chain) != 1
            or type(chain[0]) is not dict
        ):
            raise ValueError(
                f"{path.name} must carry exactly one calibration run record"
            )
        observed_record = dict(chain[0])
        if run_record is None:
            run_record = observed_record
        elif observed_record != run_record:
            raise ValueError(f"{path.name} carries a different calibration run record")
    assert run_record is not None
    return tuple(cycles), run_record


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
    #: The headline numbers -- fidelity, the two error rates, separation --
    #: model by model.  Carried here so the caller that saves them and the
    #: caller that prints them are looking at one measurement.
    summary: Mapping[str, Any] = field(default_factory=dict)

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


def _save_report(result: CalibrationRunResult) -> Path:
    """Everything a calibration leaves for a person: the numbers and the pictures.

    The numbers first.  A run that fails to draw -- a backend that cannot
    open, a font that cannot load -- has still measured what it measured, and
    what it measured is what an operator needs.
    """

    report_root = result.artifact_path.parent / "report"
    report_root.mkdir(parents=True)
    write_readable_json(report_root / "summary.json", result.summary)
    (report_root / "summary.txt").write_text(
        "\n".join(summary_lines(result.summary)) + "\n", encoding="utf-8"
    )
    _save_report_images(result, report_root)
    return report_root


def _save_report_images(result: CalibrationRunResult, report_root: Path) -> Path:
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
    generation = result.artifact_path.stem
    revision = len(result.capture.cycles)

    # Where the crop sat on the sensor: the pictures and the points drawn on
    # them are both stated there, so an operator reads the same numbers off a
    # calibration image, a camera panel and the ROI form.
    roi = calibration.frame_contract.roi_xywh
    origin_yx = (0, 0) if roi is None else (int(roi[1]), int(roi[0]))
    binning_yx = tuple(
        int(value) for value in calibration.frame_contract.binning_yx
    )
    site_map_snapshot = _snapshot(
        np.asarray(result.report["reference_average"], dtype="<f8")[np.newaxis, ...],
        signal="site_map",
        roles=(SPATIAL_Y, SPATIAL_X),
        axis_specs=_image_axis_specs(
            calibration.frame_contract.image_shape,
            "sensor_pixel_xy",
            origin_yx=origin_yx,
            binning_yx=binning_yx,
        ),
        generation=generation,
        revision=revision,
    )
    # The site map itself stays in the crop's own indices -- that is what the
    # readout slices boxes out of -- so the points are moved onto the picture
    # here, at the one place both are published together.
    overlay = ImagePointOverlay(
        revision,
        tuple(
            (
                float(origin_yx[1] + binning_yx[1] * float(centre[0])),
                float(origin_yx[0] + binning_yx[0] * float(centre[1])),
            )
            for centre in site_map.centers_xy
        ),
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
        "exposure_seconds": float(point.exposure_seconds),
        "required_external_trigger_interval_seconds": (
            None
            if point.required_external_trigger_interval_seconds is None
            else float(point.required_external_trigger_interval_seconds)
        ),
        "external_trigger_integration_start_offset_seconds": (
            None
            if point.external_trigger_integration_start_offset_seconds is None
            else float(point.external_trigger_integration_start_offset_seconds)
        ),
        "gain": float(point.gain),
        "readout_mode": point.readout_mode,
    }


def _plain(value: object) -> object:
    if value is None or type(value) in (str, bool, int, float):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("device snapshot keys must be strings")
            result[key] = _plain(item)
        return result
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
        signal_plane: object,
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
        if signal_plane is None:
            raise TypeError("signal_plane must be supplied by the runtime owner")
        self.signal_plane = signal_plane
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
        pulse_facts: Mapping[str, object],
        frames_folder: Path | None,
    ) -> tuple[
        CalibrationCapture,
        Mapping[str, object],
        SampleWriter | None,
    ]:
        count = self.request.repeats * 3
        armed = False
        firing = False
        writer: SampleWriter | None = None
        try:
            # One acquisition implementation for the whole project.  This task
            # says what it needs of it -- three frames a cycle, this many
            # cycles, the exposure its own pulse cuts them with, and the
            # geometry the bench is already set to -- and the camera
            # measurement performs it.  Written out here a second time, the
            # arm arguments, the exact read and the timeout drifted from the
            # ones every other measurement uses.
            measurement = CameraMeasurementNode(
                camera=self.camera,
                request=CameraMeasurementRequest(
                    camera_key=self.request.camera_key,
                    exposure_seconds=self.request.reference_exposure_seconds,
                    roi_xywh=_roi_xywh(self.camera.working_point()),
                    repeat=self.request.repeats,
                    frames_per_cycle=3,
                    photoelectrons=self.request.photoelectrons,
                ),
                signal_plane=self.signal_plane,
                producer=f"{self.request.camera_key}:calibration",
            )
            capture = measurement.prepare(
                owns_generation=False,
                # The capture asks this between reads, which is the only
                # place a blocking one can be interrupted: checking it
                # between CYCLES leaves the operator waiting out the
                # camera's whole timeout after pressing Stop.
                should_stop=None if context is None else context.cancel_requested,
            )
            armed = True
            actual = measurement.actual_working_point
            self._actual_working_point = actual
            arm_sequencer(self.sequencer, pulse)
            sequencer_snapshot = _sequencer_snapshot(self.sequencer)
            run_record = self._run_record(
                actual,
                sequencer_snapshot,
                pulse_facts,
            )
            if context is not None:
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
                cycle_records = capture.next_cycle()
                if cycle_records is None:
                    raise RuntimeError("calibration was cancelled")
                records = tuple(cycle_records)
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
                if frames_folder is not None:
                    if writer is None:
                        writer = SampleWriter(
                            frames_folder,
                            working_point=actual,
                            run_record=run_record,
                            generation=(
                                context.generation
                                if context is not None
                                else self.request.camera_key
                            ),
                        )
                    writer.write(len(cycles) - 1, cycle)
                if context is not None:
                    output = capture_preview_output(
                        cycle,
                        frame_shape=actual.frame_shape_yx,
                        origin_yx=actual.roi_origin_yx,
                        binning_yx=actual.binning_yx,
                        generation=context.generation,
                        revision=len(cycles),
                        run_record=run_record,
                    )
                    context.commit_live(
                        {CAPTURE_PREVIEW_DECLARATION.name: output}
                    )
                    context.report_progress(
                        "Capturing calibration",
                        current=len(cycles),
                        total=self.request.repeats,
                    )
            terminal = capture.close()
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
                writer,
            )
        except BaseException:
            if armed:
                capture.close()
            if firing:
                self._safe()
            raise

    def _replay_saved_frames(
        self,
        context: object | None,
    ) -> tuple[CalibrationCapture, dict[str, object], FrameContract]:
        """Calibrate a folder of saved samples, with no bench involved.

        The same protocol read back: three frames a sample, and the geometry
        they were TAKEN with, which is recorded in the archive beside them.
        Reading the crop off the current camera instead would place the sites
        on the wrong pixels the moment an ROI had moved since -- and the point
        of saving frames is that they can be calibrated again later, with
        other detection settings, when the bench has moved on.

        And it publishes the same dataset an acquisition publishes, sample by
        sample.  A node's declared outputs are a statement about the node, not
        about where its frames came from: publishing them only on the camera
        branch left the replay finishing with nothing published, which the
        runtime refuses -- "declared Dataset outputs but did not publish final
        outputs" -- so loading a folder could not be made to work at all.
        """

        cycles, run_record = read_saved_samples(self.request.saved_frames_path)
        saved_request = dict(run_record.get("request") or {})
        camera_key = str(saved_request.get("camera_key") or self.request.camera_key)
        devices = dict(run_record.get("actual_devices") or {})
        camera = devices.get(camera_key)
        if not isinstance(camera, Mapping):
            raise ValueError(
                f"the saved frames record no working point for camera "
                f"{camera_key!r}, so the crop they were taken with is unknown"
            )
        frame_shape = tuple(int(value) for value in camera["frame_shape_yx"])
        sensor_shape = camera.get("sensor_shape_yx")
        roi = camera.get("roi_xywh")
        contract = FrameContract(
            (frame_shape[0], frame_shape[1]),
            sensor_shape=(
                None
                if sensor_shape is None
                else (int(sensor_shape[0]), int(sensor_shape[1]))
            ),
            roi_xywh=(
                None
                if roi is None
                else tuple(int(value) for value in roi)  # type: ignore[arg-type]
            ),
            binning_yx=tuple(int(value) for value in camera["binning_yx"]),
            exposure_seconds=float(camera["exposure_seconds"]),
            camera_id=camera_key,
            readout_mode=camera.get("readout_mode"),
        )
        if context is not None:
            for index, cycle in enumerate(cycles):
                if context.cancel_requested():
                    raise RuntimeError("calibration was cancelled")
                output = capture_preview_output(
                    cycle,
                    frame_shape=contract.image_shape,
                    origin_yx=(
                        (0, 0) if roi is None else (int(roi[1]), int(roi[0]))
                    ),
                    binning_yx=contract.binning_yx,
                    generation=context.generation,
                    revision=index + 1,
                    run_record=run_record,
                )
                context.commit_live(
                    {CAPTURE_PREVIEW_DECLARATION.name: output}
                )
                context.report_progress(
                    "Reading saved frames",
                    current=index + 1,
                    total=len(cycles),
                )
        terminal = CameraCaptureTerminalRecord(len(cycles) * 3, True, True, True)
        return CalibrationCapture(cycles, terminal), dict(run_record), contract

    def _frame_contract(
        self,
        actual: CameraWorkingPoint,
        pulse: Mapping[str, object],
    ) -> FrameContract:
        roi_x, roi_y, roi_width, roi_height = _roi_xywh(actual)
        return FrameContract(
            actual.frame_shape_yx,
            sensor_shape=actual.sensor_shape_yx,
            roi_xywh=(roi_x, roi_y, roi_width, roi_height),
            binning_yx=actual.binning_yx,
            # What the SENSOR integrated, read back from the sensor: the one
            # witness to the condition these thresholds were measured under.
            # It used to be the smaller of that and the readout window the
            # pulse gates the probe light with -- a third number that is
            # neither, and one every downstream node then armed its camera at.
            # A calibration says what it did; it does not legislate the next
            # run, whose physics is the operator's to judge.
            exposure_seconds=actual.exposure_seconds,
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
            detection_sigma=self.request.detection_sigma,
        )

    def _run(self, context: object | None) -> CalibrationRunResult:
        self._actual_working_point = None
        self._result = None
        try:
            # The run's own folder is named BEFORE anything is acquired, so
            # the frames can be written into it as they arrive rather than
            # after a result exists.  A run that never produces one still
            # leaves every sample it paid for.
            #
            # Everything a run leaves lives INSIDE that folder, the artifact
            # included: one calibration is one directory an operator can
            # copy, move or delete whole, rather than a file that has to be
            # kept beside a folder of the same name.
            run_folder = unique_path(self.artifact_directory, "calibration", "")
            artifact_path = run_folder / f"{run_folder.name}.json"
            writer: SampleWriter | None = None
            if self.request.frame_source == FRAMES_FROM_FOLDER:
                if context is not None:
                    context.report_progress("Reading saved frames")
                capture, run_record, contract = self._replay_saved_frames(context)
                pulse_facts: Mapping[str, object] = dict(
                    run_record.get("pulse") or {}
                )
            else:
                pulse = self._resolve_pulse()
                pulse_facts = self._pulse_facts(pulse)
                capture, run_record, writer = self._capture(
                    pulse,
                    context=context,
                    pulse_facts=pulse_facts,
                    frames_folder=(
                        run_folder / "frames"
                        if self.request.save_frames
                        else None
                    ),
                )
                actual = self._actual_working_point
                if not isinstance(actual, CameraWorkingPoint):
                    raise TypeError(
                        "the camera measurement must report its working point"
                    )
                contract = self._frame_contract(actual, pulse_facts)
            if writer is not None:
                # The pictures, now that the camera is no longer waiting on
                # this thread for its next trigger.
                if context is not None:
                    context.report_progress("Drawing saved frames")
                writer.render()
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
            if context is not None:
                # Analysis remains cancellable.  Once artifact publication
                # begins, Stop must not leave a successful-looking half report
                # or relabel durable output as a cancelled run.
                context.seal_terminal()
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
                readout_summary(analysis, run_chain=(run_record,)),
            )
            if context is not None:
                context.report_progress("Saving calibration report")
            _save_report(result)
            self._result = result
            if context is not None:
                # What it found, not just that it finished: the operator's
                # first question is which model to read out with and how well
                # it does, and the answer is measured by the time we are here.
                for line in summary_lines(result.summary):
                    context.report_progress(line)
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
