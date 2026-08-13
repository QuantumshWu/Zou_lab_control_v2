"""Minimal same-shot occupancy processor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from zlc_data import (
    OwnedSnapshot,
    PointColumn,
    SITE,
    SPATIAL_X,
    SPATIAL_Y,
)
from zlc_runtime import DatasetCoverage
from zlc_runtime import DatasetOutputDeclaration, LiveDatasetOutput
from zlc_runtime import SignalValue

from zlc_atom.data import snapshot_from_array
from zlc_atom.nodes.calibration import ReadoutModelKind, TrapCalibration
from zlc_atom.nodes.calibration.calibration import classify_threshold


#: The per-site, per-frame judgement an image panel annotates itself with.
#: Named here because the console picks its overlay candidates by contract and
#: must not learn a second spelling of this one.
SITE_STATUS_CONTRACT = "occupancy.occupied.v1"

_OUTPUT_DECLARATIONS = (
    DatasetOutputDeclaration("counts", "occupancy.counts.v1"),
    DatasetOutputDeclaration("occupied", SITE_STATUS_CONTRACT),
    DatasetOutputDeclaration("valid", "occupancy.valid.v1"),
    DatasetOutputDeclaration("rate", "occupancy.rate.v1"),
    DatasetOutputDeclaration("frame_judged", "occupancy.frame_judged.v1"),
)


@dataclass(frozen=True)
class OccupancyResult:
    #: All four share the parent's ``(repeat, point)`` leading pair: this
    #: classifier judges one frame at a time and changes no point set, so it
    #: INHERITS the point axis of the frames it read.  Sites are cell data --
    #: an image resampled onto the trap lattice, not something anyone scanned.
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
            counts.ndim != 3
            or counts.shape != occupied.shape
            or counts.shape != valid.shape
            or rate.shape != counts.shape[:2]
            or frame_judged.ndim != 4
            or frame_judged.shape[:2] != counts.shape[:2]
        ):
            raise ValueError(
                "occupancy outputs are (repeat, point, sites) beside "
                "(repeat, point) and (repeat, point, y, x)"
            )
        for value in (counts, occupied, valid, rate, frame_judged):
            value.setflags(write=False)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "occupied", occupied)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "rate", rate)
        object.__setattr__(self, "frame_judged", frame_judged)
        object.__setattr__(self, "artifacts", dict(self.artifacts))


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
        #: The calibration placed against the crop the RUN is taking, once a
        #: run record says what that crop is.  Until then a run is assumed to
        #: be taking the crop the calibration was measured on.
        self._runtime: TrapCalibration | None = None
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

    def _source_point_column(self, snapshot: OwnedSnapshot) -> PointColumn:
        """Read the parent's declared cycle: cell axes are (y, x), points vary.

        Read, never sniffed.  This used to branch on ``ndim`` and on whether
        ``shape[1] == 1``, which made a one-frame cycle silently lose its frame
        point axis -- the shape of an array cannot say what its axes MEAN, and
        a schema already says it.
        """

        schema = snapshot.block.schema
        axes = schema.cell_schema.data_axes
        if tuple(axis.role for axis in axes) != (SPATIAL_Y, SPATIAL_X):
            raise ValueError(
                "occupancy frames must declare exactly SPATIAL_Y, SPATIAL_X cell axes"
            )
        observed = tuple(int(axis.size) for axis in axes)
        expected = self.readout.frame_contract.image_shape
        if observed != expected:
            raise ValueError(
                f"frame shape {observed} differs from the crop this readout "
                f"is placed against {expected}"
            )
        columns = schema.point_table.columns
        if len(columns) != 1:
            raise ValueError(
                "occupancy inherits its parent's point column and the source "
                f"declares {len(columns)}"
            )
        return columns[0]

    @property
    def readout(self) -> TrapCalibration:
        """The calibration as it applies to the frames actually arriving."""

        return self._runtime if self._runtime is not None else self.calibration

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

        # A run may crop the sensor differently from the calibration: where a
        # trap IS, is a fact about the sensor, so a different ROI numbers the
        # same places differently and the calibration can be read against it.
        # Only a crop that does not COVER the sites is refused, and only by
        # the calibration itself -- it owns both crops, so it owns the
        # translation.  Binning is refused there too: it changes what a pixel
        # means, and with it every threshold measured in pixels.
        origin = pair("roi_origin_yx")
        shape = pair("roi_shape_yx")
        binning = pair("binning_yx") or tuple(contract.binning_yx)
        if origin is not None and shape is not None:
            roi = (int(origin[1]), int(origin[0]), int(shape[1]), int(shape[0]))
            frame = (
                int(shape[0]) // int(binning[0]),
                int(shape[1]) // int(binning[1]),
            )
            self._runtime = self.calibration.rebased(roi, binning, frame)
        elif tuple(binning) != tuple(contract.binning_yx):
            raise ValueError(
                f"camera binning {tuple(binning)} differs from calibration "
                f"{tuple(contract.binning_yx)}"
            )

    @property
    def dataset_output_declarations(self) -> tuple[DatasetOutputDeclaration, ...]:
        return _OUTPUT_DECLARATIONS

    def signal_key(self, output_name: str) -> str:
        names = {declaration.name for declaration in _OUTPUT_DECLARATIONS}
        if str(output_name) not in names:
            raise KeyError(f"unknown occupancy output {output_name!r}")
        return f"@logic/{self.instance_id}/{output_name}"

    def process(
        self,
        frames: OwnedSnapshot,
        *,
        generation: str,
        revision: int,
    ) -> OccupancyResult:
        """Derive occupancy from a published cycle, stamped with its run.

        ``frames`` is the parent's SNAPSHOT, not a bare array: the point axis
        this derivation inherits is a schema fact, and there is no way to
        recover a point column from a block of numbers.

        The stamps are INHERITED rather than invented: a derived signal that
        counted independently would drift out of step with the frames it
        describes, and the same-shot family would no longer line up.

        Required, with no default.  They defaulted to a constant that only the
        offline callers ever accepted -- the same constant ``snapshot_from_array``
        had its defaults removed for, because a publication stamped generation
        "0" forever freezes every live plot downstream, which rejects a revision
        that is not newer than the one it holds.
        """

        if not isinstance(frames, OwnedSnapshot):
            raise TypeError("occupancy process requires zlc_data.OwnedSnapshot")
        point_column = self._source_point_column(frames)
        point_role = point_column.role
        images = np.asarray(frames.block.values)
        repeats, points = images.shape[0], images.shape[1]
        n_sites = self.readout.n_sites
        flat = images.reshape((repeats * points, *images.shape[2:]))
        # Extracted once.  detect() begins by calling signals(), so asking for
        # counts and then for occupancy ran the box or PSF extraction over every
        # site of every frame twice -- for a classification that is one
        # comparison against the thresholds the calibration already carries.
        counts = np.asarray(
            [
                self.readout.signals(image, model_kind=self.model.kind)
                for image in flat
            ],
            dtype="<f8",
        )
        model = self.model
        site_valid = (
            self.readout.site_map.valid_sites
            & model.usable_sites
            & np.isfinite(model.thresholds)
        )
        counts[:, ~site_valid] = np.nan
        occupied = classify_threshold(counts, model.thresholds)
        valid = np.broadcast_to(site_valid, counts.shape).copy()
        counts = counts.reshape((repeats, points, n_sites))
        occupied = occupied.reshape((repeats, points, n_sites))
        valid = valid.reshape((repeats, points, n_sites))
        valid_count = np.sum(valid, axis=-1)
        rate = np.divide(
            np.sum(occupied & valid, axis=-1, dtype=float),
            valid_count,
            out=np.full(valid_count.shape, np.nan, dtype="<f8"),
            where=valid_count > 0,
        )
        site_axis = self.readout.site_map.site_axis
        artifacts = {
            "counts": snapshot_from_array(
                counts,
                producer=self.producer,
                signal="counts",
                roles=(point_role, SITE),
                axis_specs={SITE: site_axis},
                point_columns={point_role: point_column},
                generation=generation,
                revision=revision,
            ),
            "occupied": snapshot_from_array(
                occupied,
                producer=self.producer,
                signal="occupied",
                roles=(point_role, SITE),
                axis_specs={SITE: site_axis},
                point_columns={point_role: point_column},
                generation=generation,
                revision=revision,
            ),
            "valid": snapshot_from_array(
                valid,
                producer=self.producer,
                signal="valid",
                roles=(point_role, SITE),
                axis_specs={SITE: site_axis},
                point_columns={point_role: point_column},
                generation=generation,
                revision=revision,
            ),
            "rate": snapshot_from_array(
                rate,
                producer=self.producer,
                signal="rate",
                roles=(point_role,),
                point_columns={point_role: point_column},
                generation=generation,
                revision=revision,
            ),
            "frame_judged": snapshot_from_array(
                images,
                producer=self.producer,
                signal="frame_judged",
                roles=(point_role, SPATIAL_Y, SPATIAL_X),
                point_columns={point_role: point_column},
                generation=generation,
                revision=revision,
            ),
        }
        return OccupancyResult(
            counts,
            occupied,
            valid,
            rate,
            images,
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
        snapshot = signal_value.snapshot
        self._source_point_column(snapshot)
        self._validate_source_run_record(signal_value)
        result = self.process(snapshot, **inherited_stamps(snapshot))
        return self._live_outputs(
            result,
            frames_signal=self.source_signal or signal_value.name,
        )

__all__ = ["SITE_STATUS_CONTRACT", "OccupancyProcessor", "OccupancyResult"]
