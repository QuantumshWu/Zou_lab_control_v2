"""The live camera preview published while Calibration is running."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

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
    DataBlock,
    DatasetComponentValidity,
    DatasetSchema,
    OwnedSnapshot,
    ValidityContract,
    ValueSchema,
)
from zlc_runtime import (
    DatasetOutputDeclaration,
    LiveDatasetOutput,
    MonitorCoverage,
)

from zlc_atom.data import snapshot_from_array
from zlc_atom.devices.camera.contract import CameraFrameRecord

CAPTURE_PREVIEW_DECLARATION = DatasetOutputDeclaration(
    "capture_preview", "calibration.capture-preview.v1"
)

#: One calibration acquisition fires three camera windows -- long, readout,
#: long -- and they are three POINTS of the cycle, not three looks at the same
#: thing: each is exposed at a different place in the pulse, so their physics
#: differs.  The preview used to keep ``images[-1]`` and drop the other two,
#: which threw away the readout frame the whole calibration is about.
_PREVIEW_FRAMES = 3


def _image_axis_specs(
    frame_shape: tuple[int, int],
    coordinate_frame: str | CoordinateFrameId,
    *,
    origin_yx: tuple[int, int] = (0, 0),
    binning_yx: tuple[int, int] = (1, 1),
) -> dict[AxisRoleId, AxisSpec]:
    """One image schema for Calibration live and FINAL publications.

    The pictures a calibration shows are crops of the same sensor the camera
    panel shows, so they are labelled the same way: in the sensor's own
    pixels.  Indexed from zero instead, the same trap read as two different
    places depending on which ROI the run happened to use, and two pictures of
    one lattice side by side disagreed about where it was.
    """

    height, width = (int(value) for value in frame_shape)
    origin_y, origin_x = (int(value) for value in origin_yx)
    step_y, step_x = (max(1, int(value)) for value in binning_yx)
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
            coordinates=tuple(origin_y + step_y * index for index in range(height)),
            unit="pixel",
            coordinate_frame=frame,
        ),
        SPATIAL_X: AxisSpec(
            AxisId("calibration.image.x"),
            "x",
            SPATIAL_X,
            width,
            coordinates=tuple(origin_x + step_x * index for index in range(width)),
            unit="pixel",
            coordinate_frame=frame,
        ),
    }


def _generation_text(value: object) -> str:
    text = str(getattr(value, "value", value)).strip()
    if not text or text == "None":
        raise ValueError("calibration Dataset generation must be available")
    return text


def _frame_point_column(frames: int) -> PointColumn:
    """The frame-index point column one calibration cycle publishes."""

    return PointColumn(
        AxisId("calibration.capture_preview.frame"),
        "frame",
        READOUT_EVENT,
        PointColumn.NUMERIC,
        tuple(range(int(frames))),
    )


def _with_component_validity(
    snapshot: OwnedSnapshot,
    axis_ids: tuple[AxisId, ...],
    mask: object,
) -> OwnedSnapshot:
    """Restate one snapshot's validity over named CELL axes.

    Whether a site is usable is a per-site fact.  Once SITE is a cell data
    axis, ``CellValidity`` cannot carry it: that mask is one flag per
    ``(repeat, point)`` cell, i.e. one flag covering every site at once.  The
    dataset already has the contract for this; it only has to be declared.
    """

    source = snapshot.block
    cell = source.schema.cell_schema
    schema = DatasetSchema(
        source.schema.repeat_axis,
        source.schema.point_table,
        source.schema.grid_topology,
        ValueSchema(
            cell.data_axes,
            ValidityContract.components(*axis_ids),
            cell.dtype,
            cell.value_unit,
        ),
    )
    block = DataBlock(
        source.block_id,
        source.revision,
        source.values,
        DatasetComponentValidity(axis_ids, np.asarray(mask, dtype=bool)),
        schema,
    )
    return OwnedSnapshot(block.ref(snapshot.ref.stream_generation), block)


def _snapshot(
    values: object,
    *,
    signal: str,
    roles: Sequence[AxisRoleId],
    generation: object,
    revision: int,
    validity_axis_ids: tuple[AxisId, ...] | None = None,
    validity_mask: object | None = None,
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
    if (validity_axis_ids is None) != (validity_mask is None):
        raise ValueError("component validity needs both its axes and its mask")
    if validity_axis_ids is not None:
        snapshot = _with_component_validity(
            snapshot, validity_axis_ids, validity_mask
        )
    return snapshot


def _cycle_array(
    cycle: Sequence[CameraFrameRecord],
    frame_shape: tuple[int, int],
) -> np.ndarray:
    """Three frames as one array, in whatever the frames themselves are.

    It used to be cast to the WORKING POINT's dtype, which is the sensor's
    pixel format -- true of what the camera reads out, and false of what this
    node was handed the moment a run asked for photoelectrons: 0.535 e- came
    back as 0, and a pixel below the sensor's offset came back as 65535.  A
    frame's dtype is the frame's own fact.
    """

    records = tuple(cycle)
    if len(records) != _PREVIEW_FRAMES or any(
        not isinstance(record, CameraFrameRecord) for record in records
    ):
        raise ValueError("calibration preview cycle must contain three frame records")
    images = np.stack([np.asarray(record.image) for record in records], axis=0)
    if images.shape != (_PREVIEW_FRAMES, *frame_shape):
        raise ValueError("calibration preview frame shape changed during capture")
    return images


def cycle_snapshot(
    cycle: Sequence[CameraFrameRecord],
    *,
    frame_shape: tuple[int, int],
    origin_yx: tuple[int, int],
    binning_yx: tuple[int, int],
    generation: object,
    revision: int,
    value_unit: str | None,
) -> OwnedSnapshot:
    """One acquired cycle, as the dataset it is.

    THE translation, for the live preview and for the frames written to disk
    as they arrive.  Two of them would be two datasets of the same three
    frames: the picture an operator watched and the picture they reopened
    would agree about the numbers and disagree about the axes.

    The three frames are POINT ROWS of one publication -- one moment of the
    pulse each -- and the pixel axes carry the sensor pixels the crop covers,
    so a reopened file is placed on the sensor exactly as the live panel was.
    """

    return _snapshot(
        _cycle_array(cycle, frame_shape)[None, ...],
        signal=CAPTURE_PREVIEW_DECLARATION.name,
        roles=(READOUT_EVENT, SPATIAL_Y, SPATIAL_X),
        axis_specs=_image_axis_specs(
            frame_shape,
            "sensor_pixel_xy",
            origin_yx=origin_yx,
            binning_yx=binning_yx,
        ),
        point_columns={READOUT_EVENT: _frame_point_column(_PREVIEW_FRAMES)},
        value_unit=value_unit,
        generation=generation,
        revision=revision,
    )


def capture_preview_output(
    cycle: Sequence[CameraFrameRecord],
    *,
    frame_shape: tuple[int, int],
    origin_yx: tuple[int, int],
    binning_yx: tuple[int, int],
    generation: object,
    revision: int,
    run_record: Mapping[str, object],
    value_unit: str | None,
) -> LiveDatasetOutput:
    """Translate one complete long/readout/long cycle into its live event.

    Calibration keeps no mutable preview history.  The acquisition thread
    commits this immutable three-frame event directly; Runtime owns its
    publication identity and the Monitor keeps only the newest complete
    cycle.
    """

    snapshot = cycle_snapshot(
        cycle,
        frame_shape=frame_shape,
        origin_yx=origin_yx,
        binning_yx=binning_yx,
        generation=generation,
        revision=revision,
        value_unit=value_unit,
    )
    return LiveDatasetOutput(
        CAPTURE_PREVIEW_DECLARATION,
        snapshot,
        MonitorCoverage(_PREVIEW_FRAMES, _PREVIEW_FRAMES),
        run_record,
    )


__all__ = [
    "CAPTURE_PREVIEW_DECLARATION",
    "capture_preview_output",
    "cycle_snapshot",
]
