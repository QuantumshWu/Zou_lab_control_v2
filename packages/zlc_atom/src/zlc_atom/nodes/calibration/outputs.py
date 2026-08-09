"""The live camera preview published while Calibration is running."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Callable

import numpy as np
from zlc_data import (
    AxisId,
    AxisRoleId,
    AxisSpec,
    CoordinateFrameId,
    PointColumn,
    READOUT_EVENT,
    SPATIAL_X,
    SPATIAL_Y,
    CellValidity,
    DataBlock,
    OwnedSnapshot,
)
from zlc_runtime import (
    DatasetCoverage,
    DatasetOutputDeclaration,
    LiveDatasetOutput,
)

from zlc_atom.data import snapshot_from_array
from zlc_atom.devices.camera.contract import CameraFrameRecord

CAPTURE_PREVIEW_DECLARATION = DatasetOutputDeclaration(
    "capture_preview", "calibration.capture-preview.v1"
)


def _image_axis_specs(
    frame_shape: tuple[int, int],
    coordinate_frame: str | CoordinateFrameId,
) -> dict[AxisRoleId, AxisSpec]:
    """One image schema for Calibration live and FINAL publications."""

    height, width = (int(value) for value in frame_shape)
    frame = (
        coordinate_frame
        if isinstance(coordinate_frame, CoordinateFrameId)
        else CoordinateFrameId(str(coordinate_frame))
    )
    return {
        SPATIAL_Y: AxisSpec(
            AxisId("calibration.image.y"),
            "y",
            SPATIAL_Y,
            height,
            unit="pixel",
            coordinate_frame=frame,
        ),
        SPATIAL_X: AxisSpec(
            AxisId("calibration.image.x"),
            "x",
            SPATIAL_X,
            width,
            unit="pixel",
            coordinate_frame=frame,
        ),
    }


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
            axis_specs=_image_axis_specs(self.frame_shape, "image_pixel_xy"),
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


__all__ = [
    "CAPTURE_PREVIEW_DECLARATION",
    "CalibrationCapturePreviewSlot",
]
