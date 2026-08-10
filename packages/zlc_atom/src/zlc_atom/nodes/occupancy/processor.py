"""Minimal same-shot occupancy processor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from zlc_data import (
    AxisId,
    AxisRoleId,
    CoordinateFrameId,
    DataBlock,
    DatasetSchema,
    OwnedSnapshot,
    PointColumn,
    PointTable,
    READOUT_EVENT,
    SITE,
    SPATIAL_X,
    SPATIAL_Y,
)
from zlc_plot import ImagePointOverlay, PointStatus
from zlc_runtime import DatasetCoverage
from zlc_runtime import DatasetOutputDeclaration, LiveDatasetOutput
from zlc_runtime import SignalValue

from zlc_atom.devices.camera.contract import CameraFrameRecord
from zlc_atom.data import snapshot_from_array
from zlc_atom.nodes.calibration import ReadoutModelKind, TrapCalibration
from zlc_atom.nodes.calibration.calibration import classify_threshold


_OUTPUT_DECLARATIONS = (
    DatasetOutputDeclaration("counts", "occupancy.counts.v1"),
    DatasetOutputDeclaration("occupied", "occupancy.occupied.v1"),
    DatasetOutputDeclaration("valid", "occupancy.valid.v1"),
    DatasetOutputDeclaration("rate", "occupancy.rate.v1"),
    DatasetOutputDeclaration("frame_judged", "occupancy.frame_judged.v1"),
    DatasetOutputDeclaration("site_overlay", ImagePointOverlay.CONTRACT_ID),
)


@dataclass(frozen=True)
class OccupancyResult:
    counts: np.ndarray
    occupied: np.ndarray
    valid: np.ndarray
    rate: np.ndarray
    frame_judged: np.ndarray
    artifacts: dict[str, OwnedSnapshot] = field(default_factory=dict)

    def __post_init__(self) -> None:
        counts = np.asarray(self.counts, dtype="<f8")
        occupied = np.asarray(self.occupied, dtype=bool)
        valid = np.asarray(self.valid, dtype=bool)
        rate = np.asarray(self.rate, dtype="<f8")
        frame_judged = np.asarray(self.frame_judged)
        if (
            counts.shape != occupied.shape
            or counts.shape != valid.shape
            or counts.ndim < 2
            or rate.shape != counts.shape[:-1]
            or frame_judged.ndim < 3
            or frame_judged.shape[:-2] != counts.shape[:-1]
        ):
            raise ValueError("occupancy outputs must share leading dimensions")
        for value in (counts, occupied, valid, rate, frame_judged):
            value.setflags(write=False)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "occupied", occupied)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "rate", rate)
        object.__setattr__(self, "frame_judged", frame_judged)
        object.__setattr__(self, "artifacts", dict(self.artifacts))


def _image(frame: object) -> np.ndarray:
    if isinstance(frame, CameraFrameRecord):
        return np.asarray(frame.image)
    return np.asarray(frame.values if hasattr(frame, "values") else frame)


def _unpack_frames(frames: object) -> tuple[list[object], tuple[int, ...]]:
    if hasattr(frames, "frames") and not isinstance(frames, (np.ndarray, list, tuple)):
        frames = getattr(frames, "frames")
    if hasattr(frames, "values") and not isinstance(frames, (np.ndarray, list, tuple)):
        frames = getattr(frames, "values")
    if isinstance(frames, np.ndarray):
        array = np.asarray(frames)
        if array.ndim == 5 and array.shape[1] == 1:
            array = array[:, 0]
        if array.ndim == 3:
            return [array[index] for index in range(array.shape[0])], (int(array.shape[0]),)
        if array.ndim == 4:
            if array.shape[1] == 1:
                return [array[index, 0] for index in range(array.shape[0])], (int(array.shape[0]),)
            return [array[repeat, event] for repeat in range(array.shape[0]) for event in range(array.shape[1])], (
                int(array.shape[0]),
                int(array.shape[1]),
            )
        if array.ndim < 3:
            raise ValueError("camera frame arrays require a leading frame dimension")
        raise ValueError("camera frame arrays must be (frame,y,x) or (repeat,event,y,x)")
    values = list(frames)  # type: ignore[arg-type]
    if not values:
        raise ValueError("occupancy requires at least one frame")
    if isinstance(values[0], (tuple, list)):
        cycles = [tuple(cycle) for cycle in values]
        flat = [frame for cycle in cycles for frame in cycle]
        if not flat:
            raise ValueError("occupancy cycles cannot be empty")
        return flat, (len(cycles), len(cycles[0]))
    return values, (len(values),)


def inherited_stamps(snapshot: object) -> dict[str, object]:
    """The run stamps a derived signal takes from the data it describes.

    A derived signal that counted independently would drift out of step with its
    parent, and the same-shot family would stop lining up.

    Public because ``process`` requires them: an offline caller holds the
    publication its frames came from and must say which run it is deriving.
    """

    ref = snapshot.ref
    return {
        "generation": str(getattr(ref.stream_generation, "value", ref.stream_generation)),
        "revision": int(getattr(ref.revision, "value", ref.revision)),
    }


def _site_overlay_snapshot(
    statuses: np.ndarray,
    *,
    calibration: TrapCalibration,
    producer: str,
    roles: tuple[AxisRoleId, ...],
    generation: str,
    revision: int,
) -> OwnedSnapshot:
    site_map = calibration.site_map
    frame = CoordinateFrameId(site_map.coordinate_frame)
    site = PointColumn(
        AxisId(f"{producer}.site_overlay.site"),
        "site",
        SITE,
        PointColumn.TEXT,
        tuple(site_map.site_ids),
        coordinate_labels=tuple(
            str(index) for index in range(1, site_map.n_sites + 1)
        ),
    )
    base = snapshot_from_array(
        statuses,
        producer=producer,
        signal="site_overlay",
        roles=roles,
        point_columns={SITE: site},
        generation=generation,
        revision=revision,
    )
    centers = np.asarray(site_map.centers_xy, dtype=float)
    table = PointTable(
        site_map.n_sites,
        (
            site,
            PointColumn(
                AxisId(f"{producer}.site_overlay.x"),
                "x",
                SPATIAL_X,
                PointColumn.NUMERIC,
                tuple(float(value) for value in centers[:, 0]),
                unit="pixel",
                coordinate_frame=frame,
            ),
            PointColumn(
                AxisId(f"{producer}.site_overlay.y"),
                "y",
                SPATIAL_Y,
                PointColumn.NUMERIC,
                tuple(float(value) for value in centers[:, 1]),
                unit="pixel",
                coordinate_frame=frame,
            ),
        ),
    )
    source = base.block
    schema = DatasetSchema(
        source.schema.repeat_axis,
        table,
        source.schema.grid_topology,
        source.schema.cell_schema,
    )
    block = DataBlock(
        source.block_id,
        source.revision,
        source.values,
        source.validity,
        schema,
    )
    return OwnedSnapshot(block.ref(base.ref.stream_generation), block)


class OccupancyProcessor:
    """Evaluate the calibrated readout once per camera frame."""

    def __init__(
        self,
        calibration: TrapCalibration,
        *,
        calibration_path: str | Path | None = None,
        producer: str = "occupancy",
        source_signal: str | None = None,
        model_kind: ReadoutModelKind | None = None,
    ) -> None:
        if not isinstance(calibration, TrapCalibration):
            raise TypeError("calibration must be TrapCalibration")
        if calibration.site_map.coordinate_frame != "image_pixel_xy":
            raise ValueError(
                "occupancy requires calibration centers in image_pixel_xy coordinates"
            )
        self.calibration = calibration
        self.model = calibration.select_model(model_kind)
        self.calibration_path = (
            None
            if calibration_path is None
            else Path(calibration_path).expanduser().resolve()
        )
        self.instance_id = str(producer).strip()
        if not self.instance_id:
            raise ValueError("producer must be non-empty")
        self.producer = self.instance_id
        self.source_signal = None if source_signal is None else str(source_signal).strip()

    def _validate_source_contract(self, source: SignalValue) -> None:
        axes = source.axes
        if len(axes) < 2 or tuple(axis.role for axis in axes[-2:]) != (
            SPATIAL_Y,
            SPATIAL_X,
        ):
            raise ValueError(
                "occupancy frames must end in SPATIAL_Y, SPATIAL_X axes"
            )
        observed = tuple(int(axis.size) for axis in axes[-2:])
        expected = self.calibration.frame_contract.image_shape
        if observed != expected:
            raise ValueError(
                f"frame shape {observed} differs from calibration {expected}"
            )
        self._validate_source_run_record(source)

    def _validate_source_run_record(self, source: SignalValue) -> None:
        """Check only structural camera facts present on the parent."""

        record = source.run_record
        contract = self.calibration.frame_contract
        snapshots = record.get("device_snapshots")
        if snapshots is None:
            return
        if not isinstance(snapshots, Mapping):
            raise ValueError("camera run record device_snapshots must be a mapping")
        actual = snapshots.get("camera")
        if actual is None:
            return
        if not isinstance(actual, Mapping):
            raise ValueError("camera device snapshot must be a mapping")

        def pair(name: str) -> tuple[int, int] | None:
            value = actual.get(name)
            if value is None:
                return None
            try:
                result = tuple(int(item) for item in value)  # type: ignore[arg-type]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"camera device snapshot {name} must contain two integers"
                ) from exc
            if len(result) != 2:
                raise ValueError(
                    f"camera device snapshot {name} must contain two integers"
                )
            return result

        sensor = pair("sensor_shape_yx")
        if contract.sensor_shape is not None and sensor is not None:
            if sensor != contract.sensor_shape:
                raise ValueError(
                    f"camera sensor shape {sensor} differs from calibration "
                    f"{contract.sensor_shape}"
                )

        if contract.roi_xywh is not None:
            x, y, width, height = contract.roi_xywh
            origin = pair("roi_origin_yx")
            if origin is not None and origin != (y, x):
                raise ValueError(
                    f"camera ROI origin {origin} differs from calibration {(y, x)}"
                )
            shape = pair("roi_shape_yx")
            if shape is not None and shape != (height, width):
                raise ValueError(
                    f"camera ROI shape {shape} differs from calibration "
                    f"{(height, width)}"
                )

        binning = pair("binning_yx")
        if binning is not None and binning != contract.binning_yx:
            raise ValueError(
                f"camera binning {binning} differs from calibration "
                f"{contract.binning_yx}"
            )

    @property
    def dataset_output_declarations(self) -> tuple[DatasetOutputDeclaration, ...]:
        return _OUTPUT_DECLARATIONS

    def signal_key(self, output_name: str) -> str:
        names = {declaration.name for declaration in _OUTPUT_DECLARATIONS}
        if str(output_name) not in names:
            raise KeyError(f"unknown occupancy output {output_name!r}")
        return f"@logic/{self.instance_id}/{output_name}"

    def process(self, frames: object, *, generation: str, revision: int) -> OccupancyResult:
        """Derive occupancy, stamped with the run its frames came from.

        The stamps are INHERITED rather than invented: a derived signal that
        counted independently would drift out of step with the frames it
        describes, and the same-shot family would no longer line up.

        Required, with no default.  They defaulted to a constant that only the
        offline callers ever accepted -- the same constant ``snapshot_from_array``
        had its defaults removed for, because a publication stamped generation
        "0" forever freezes every live plot downstream, which rejects a revision
        that is not newer than the one it holds.
        """

        values, leading = _unpack_frames(frames)
        # Extracted once.  detect() begins by calling signals(), so asking for
        # counts and then for occupancy ran the box or PSF extraction over every
        # site of every frame twice -- for a classification that is one
        # comparison against the thresholds the calibration already carries.
        images = [_image(frame) for frame in values]
        counts = np.asarray(
            [
                self.calibration.signals(image, model_kind=self.model.kind)
                for image in images
            ],
            dtype="<f8",
        )
        model = self.model
        site_valid = (
            self.calibration.site_map.valid_sites
            & model.usable_sites
            & np.isfinite(model.thresholds)
        )
        counts[:, ~site_valid] = np.nan
        occupied = classify_threshold(counts, model.thresholds)
        valid = np.broadcast_to(site_valid, counts.shape).copy()
        counts = counts.reshape((*leading, self.calibration.n_sites))
        occupied = occupied.reshape((*leading, self.calibration.n_sites))
        valid = valid.reshape((*leading, self.calibration.n_sites))
        statuses = np.full(valid.shape, PointStatus.UNKNOWN.code, dtype=np.uint8)
        statuses[~valid] = PointStatus.INVALID.code
        statuses[valid & ~occupied] = PointStatus.EMPTY.code
        statuses[valid & occupied] = PointStatus.OCCUPIED.code
        valid_count = np.sum(valid, axis=-1)
        rate = np.divide(
            np.sum(occupied & valid, axis=-1, dtype=float),
            valid_count,
            out=np.full(valid_count.shape, np.nan, dtype="<f8"),
            where=valid_count > 0,
        )
        frame_judged = np.stack(images, axis=0).reshape(
            (*leading, *self.calibration.frame_contract.image_shape)
        )
        site_roles = (READOUT_EVENT,) * max(counts.ndim - 2, 0) + (SITE,)
        rate_roles = (READOUT_EVENT,) * max(rate.ndim - 1, 0)
        frame_roles = (READOUT_EVENT,) * max(frame_judged.ndim - 3, 0) + (
            SPATIAL_Y,
            SPATIAL_X,
        )
        artifacts = {
            "counts": snapshot_from_array(
                counts,
                producer=self.producer,
                signal="counts",
                roles=site_roles,
                generation=generation,
                revision=revision,
            ),
            "occupied": snapshot_from_array(
                occupied,
                producer=self.producer,
                signal="occupied",
                roles=site_roles,
                generation=generation,
                revision=revision,
            ),
            "valid": snapshot_from_array(
                valid,
                producer=self.producer,
                signal="valid",
                roles=site_roles,
                generation=generation,
                revision=revision,
            ),
            "rate": snapshot_from_array(
                rate,
                producer=self.producer,
                signal="rate",
                roles=rate_roles,
                generation=generation,
                revision=revision,
            ),
            "frame_judged": snapshot_from_array(
                frame_judged,
                producer=self.producer,
                signal="frame_judged",
                roles=frame_roles,
                generation=generation,
                revision=revision,
            ),
            "site_overlay": _site_overlay_snapshot(
                statuses,
                calibration=self.calibration,
                producer=self.producer,
                roles=site_roles,
                generation=generation,
                revision=revision,
            ),
        }
        return OccupancyResult(
            counts,
            occupied,
            valid,
            rate,
            frame_judged,
            artifacts=artifacts,
        )

    def _live_outputs(
        self,
        result: OccupancyResult,
        *,
        frames_signal: str,
    ) -> dict[str, LiveDatasetOutput]:
        run_record = {
            "node": self.instance_id,
            "parameters": {
                "frames_signal": str(frames_signal),
                "calibration_path": (
                    None
                    if self.calibration_path is None
                    else str(self.calibration_path)
                ),
                "model_kind": self.model.kind.value,
            },
        }
        outputs: dict[str, LiveDatasetOutput] = {}
        for declaration in _OUTPUT_DECLARATIONS:
            snapshot = result.artifacts[declaration.name]
            total = snapshot.block.schema.repeat_axis.size * snapshot.block.schema.point_table.row_count
            outputs[declaration.name] = LiveDatasetOutput(
                declaration,
                snapshot,
                DatasetCoverage(total, total),
                run_record,
            )
        return outputs

    def evaluate(self, signal_value: SignalValue) -> dict[str, LiveDatasetOutput]:
        if not isinstance(signal_value, SignalValue):
            raise TypeError("occupancy evaluate requires zlc_runtime.SignalValue")
        self._validate_source_contract(signal_value)
        result = self.process(signal_value, **inherited_stamps(signal_value.snapshot))
        return self._live_outputs(
            result,
            frames_signal=self.source_signal or signal_value.name,
        )

__all__ = ["OccupancyProcessor", "OccupancyResult"]
