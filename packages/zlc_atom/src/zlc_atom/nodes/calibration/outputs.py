"""Typed live and FINAL Dataset outputs owned by Calibration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Callable

import numpy as np
from zlc_data import (
    AxisId,
    AxisRoleId,
    AxisSpec,
    COMPONENT,
    CoordinateFrameId,
    PointColumn,
    READOUT_EVENT,
    SITE,
    SPATIAL_X,
    SPATIAL_Y,
    CellValidity,
    DataBlock,
    OwnedSnapshot,
)
from zlc_runtime import (
    DatasetCoverage,
    DatasetOutputDeclaration,
    FinalDatasetOutput,
    LiveDatasetOutput,
)

from zlc_atom.data import snapshot_from_array
from zlc_atom.devices.camera.contract import CameraFrameRecord

from .calibration import TrapCalibration


CAPTURE_PREVIEW_DECLARATION = DatasetOutputDeclaration(
    "capture_preview", "calibration.capture-preview.v1"
)
SITE_MAP_DECLARATION = DatasetOutputDeclaration(
    "site_map", "calibration.site-map.v1"
)
FIDELITY_SITE_DECLARATION = DatasetOutputDeclaration(
    "fidelity_site", "calibration.site-fidelity.v1"
)
FIDELITY_CENTERS_DECLARATION = DatasetOutputDeclaration(
    "fidelity_centers", "calibration.site-centers.v1"
)
READOUT_SAMPLES_DECLARATION = DatasetOutputDeclaration(
    "readout_samples", "calibration.readout-samples.v1"
)
FIDELITY_THRESHOLD_DECLARATION = DatasetOutputDeclaration(
    "fidelity_threshold", "calibration.site-threshold.v1"
)
CALIBRATION_DATASET_DECLARATIONS = (
    CAPTURE_PREVIEW_DECLARATION,
    SITE_MAP_DECLARATION,
    FIDELITY_SITE_DECLARATION,
    FIDELITY_CENTERS_DECLARATION,
    READOUT_SAMPLES_DECLARATION,
    FIDELITY_THRESHOLD_DECLARATION,
)


def _generation_text(value: object) -> str:
    text = str(getattr(value, "value", value)).strip()
    if not text or text == "None":
        raise ValueError("calibration Dataset generation must be available")
    return text


def _with_cell_validity(
    snapshot: OwnedSnapshot,
    mask: object,
) -> OwnedSnapshot:
    validity = CellValidity(np.asarray(mask, dtype=bool))
    block = DataBlock(
        snapshot.block.block_id,
        snapshot.block.revision,
        snapshot.block.values,
        validity,
        snapshot.block.schema,
    )
    return OwnedSnapshot(block.ref(snapshot.ref.stream_generation), block)


def _snapshot(
    values: object,
    *,
    signal: str,
    roles: Sequence[AxisRoleId],
    generation: object,
    revision: int,
    cell_validity: object | None = None,
    axis_specs: Mapping[AxisRoleId, AxisSpec] | None = None,
    point_columns: Mapping[AxisRoleId, PointColumn] | None = None,
    value_unit: str | None = None,
) -> OwnedSnapshot:
    snapshot = snapshot_from_array(
        values,
        producer="calibration",
        signal=signal,
        roles=roles,
        axis_specs=axis_specs,
        point_columns=point_columns,
        value_unit=value_unit,
        generation=_generation_text(generation),
        revision=revision,
    )
    if cell_validity is not None:
        snapshot = _with_cell_validity(snapshot, cell_validity)
    return snapshot


def _cycle_array(
    cycle: Sequence[CameraFrameRecord],
    frame_shape: tuple[int, int],
    dtype: np.dtype,
) -> np.ndarray:
    records = tuple(cycle)
    if len(records) != 3 or any(
        not isinstance(record, CameraFrameRecord) for record in records
    ):
        raise ValueError("calibration preview cycle must contain three frame records")
    images = np.stack([np.asarray(record.image) for record in records], axis=0)
    if images.shape != (3, *frame_shape):
        raise ValueError("calibration preview frame shape changed during capture")
    return images.astype(dtype, copy=False)


class CalibrationCapturePreviewSlot:
    """Fixed-extent repeat x three-frame live preview for one Task run."""

    def __init__(
        self,
        *,
        repeats: int,
        frame_shape: tuple[int, int],
        dtype: object,
        generation: object,
        run_record: Mapping[str, object],
    ) -> None:
        self.repeats = int(repeats)
        if self.repeats <= 0:
            raise ValueError("calibration preview repeats must be positive")
        self.frame_shape = tuple(int(value) for value in frame_shape)
        if len(self.frame_shape) != 2 or any(value <= 0 for value in self.frame_shape):
            raise ValueError("calibration preview frame_shape must be positive Y,X")
        self.dtype = np.dtype(dtype).newbyteorder("<")
        self.generation = generation
        self.run_record = dict(run_record)
        self._values = np.zeros(
            (self.repeats, 3, *self.frame_shape), dtype=self.dtype
        )
        self._written = 0
        self._revision = 0
        self._listener: Callable[[], None] | None = None
        self._closed = False

    @property
    def written_cycles(self) -> int:
        return self._written

    def set_change_listener(self, listener: Callable[[], None]) -> None:
        if self._closed:
            raise RuntimeError("calibration preview slot is closed")
        if not callable(listener):
            raise TypeError("calibration preview listener must be callable")
        if self._listener is not None:
            raise RuntimeError("calibration preview listener is already attached")
        self._listener = listener

    def update(self, cycle: Sequence[CameraFrameRecord]) -> None:
        if self._closed:
            raise RuntimeError("calibration preview slot is closed")
        if self._listener is None:
            raise RuntimeError("calibration preview slot is not attached")
        if self._written >= self.repeats:
            raise RuntimeError("calibration preview received too many cycles")
        self._values[self._written] = _cycle_array(
            cycle, self.frame_shape, self.dtype
        )
        self._written += 1
        self._revision += 1
        self._listener()

    def freeze_live_outputs(self) -> dict[str, LiveDatasetOutput]:
        if self._closed or self._written <= 0:
            raise RuntimeError("calibration preview has no readable cycle")
        validity = np.zeros((self.repeats, 1), dtype=bool)
        validity[: self._written] = True
        snapshot = _snapshot(
            self._values,
            signal=CAPTURE_PREVIEW_DECLARATION.name,
            roles=(READOUT_EVENT, SPATIAL_Y, SPATIAL_X),
            generation=self.generation,
            revision=self._revision,
            cell_validity=validity,
        )
        return {
            CAPTURE_PREVIEW_DECLARATION.name: LiveDatasetOutput(
                CAPTURE_PREVIEW_DECLARATION,
                snapshot,
                DatasetCoverage(self._written, self.repeats),
                self.run_record,
            )
        }

    def close(self) -> None:
        self._closed = True
        self._listener = None


def calibration_final_outputs(
    *,
    calibration: TrapCalibration,
    capture_cycles: Sequence[Sequence[CameraFrameRecord]],
    report: Mapping[str, object],
    generation: object,
    run_record: Mapping[str, object],
) -> dict[str, FinalDatasetOutput]:
    """Materialize the complete preview and five report-facing FINAL outputs."""

    if not isinstance(calibration, TrapCalibration):
        raise TypeError("calibration must be TrapCalibration")
    cycles = tuple(tuple(cycle) for cycle in capture_cycles)
    if not cycles:
        raise ValueError("calibration final outputs require captured cycles")
    frame_shape = calibration.frame_contract.image_shape
    capture_values = np.stack(
        [
            _cycle_array(cycle, frame_shape, np.asarray(cycle[0].image).dtype)
            for cycle in cycles
        ],
        axis=0,
    )
    model = calibration.select_model()
    models_report = report.get("models")
    if not isinstance(models_report, Mapping):
        raise TypeError("calibration report models must be a mapping")
    model_report = models_report.get(model.kind.value)
    if not isinstance(model_report, Mapping):
        raise KeyError(model.kind.value)
    reference_average = np.asarray(report["reference_average"], dtype="<f8")
    centers = np.asarray(calibration.site_map.centers_xy, dtype="<f8")
    short_signals = np.asarray(model_report["short_signals"], dtype="<f8")
    labels_valid = np.asarray(report["labels_valid"], dtype=bool)
    site_valid = (
        calibration.site_map.valid_sites
        & model.usable_sites
        & np.isfinite(model.thresholds)
    )
    sample_valid = (
        labels_valid
        & np.isfinite(short_signals)
        & site_valid[np.newaxis, :]
    )
    coordinate_frame = CoordinateFrameId(calibration.site_map.coordinate_frame)
    site_column = PointColumn(
        AxisId("calibration.site"),
        "site",
        SITE,
        PointColumn.TEXT,
        calibration.site_map.site_ids,
    )
    site_columns = {SITE: site_column}
    image_axes = {
        SPATIAL_Y: AxisSpec(
            AxisId("calibration.image.y"),
            "y",
            SPATIAL_Y,
            frame_shape[0],
            unit="pixel",
            coordinate_frame=coordinate_frame,
        ),
        SPATIAL_X: AxisSpec(
            AxisId("calibration.image.x"),
            "x",
            SPATIAL_X,
            frame_shape[1],
            unit="pixel",
            coordinate_frame=coordinate_frame,
        ),
    }
    center_component_axis = AxisSpec(
        AxisId("calibration.site-center.component"),
        "coordinate",
        COMPONENT,
        2,
        coordinates=("x", "y"),
        unit="pixel",
        coordinate_frame=coordinate_frame,
    )
    final_revision = len(cycles)
    snapshots = {
        "capture_preview": _snapshot(
            capture_values,
            signal="capture_preview",
            roles=(READOUT_EVENT, SPATIAL_Y, SPATIAL_X),
            axis_specs=image_axes,
            generation=generation,
            revision=final_revision,
        ),
        "site_map": _snapshot(
            reference_average[np.newaxis, ...],
            signal="site_map",
            roles=(SPATIAL_Y, SPATIAL_X),
            axis_specs=image_axes,
            generation=generation,
            revision=final_revision,
        ),
        "fidelity_site": _snapshot(
            model.quality[np.newaxis, :],
            signal="fidelity_site",
            roles=(SITE,),
            point_columns=site_columns,
            generation=generation,
            revision=final_revision,
            cell_validity=(site_valid & np.isfinite(model.quality))[np.newaxis, :],
        ),
        "fidelity_centers": _snapshot(
            centers[np.newaxis, ...],
            signal="fidelity_centers",
            roles=(SITE, COMPONENT),
            axis_specs={COMPONENT: center_component_axis},
            point_columns=site_columns,
            value_unit="pixel",
            generation=generation,
            revision=final_revision,
            cell_validity=(
                calibration.site_map.valid_sites
                & np.all(np.isfinite(centers), axis=1)
            )[np.newaxis, :],
        ),
        "readout_samples": _snapshot(
            short_signals,
            signal="readout_samples",
            roles=(SITE,),
            point_columns=site_columns,
            generation=generation,
            revision=final_revision,
            cell_validity=sample_valid,
        ),
        "fidelity_threshold": _snapshot(
            model.thresholds[np.newaxis, :],
            signal="fidelity_threshold",
            roles=(SITE,),
            point_columns=site_columns,
            generation=generation,
            revision=final_revision,
            cell_validity=site_valid[np.newaxis, :],
        ),
    }
    declarations = {
        declaration.name: declaration
        for declaration in CALIBRATION_DATASET_DECLARATIONS
    }
    if tuple(snapshots) != tuple(declarations):
        raise RuntimeError("calibration final output order changed")
    return {
        name: FinalDatasetOutput(declarations[name], snapshot, run_record)
        for name, snapshot in snapshots.items()
    }


__all__ = [
    "CALIBRATION_DATASET_DECLARATIONS",
    "CAPTURE_PREVIEW_DECLARATION",
    "CalibrationCapturePreviewSlot",
    "calibration_final_outputs",
]
