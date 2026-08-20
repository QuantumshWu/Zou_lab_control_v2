"""Minimal same-shot occupancy processor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
from zlc_data import (
    DatasetSchema,
    OwnedSnapshot,
    PointColumn,
    SPATIAL_X,
    SPATIAL_Y,
    ValidityContract,
    ValueSchema,
    owned_snapshot_from_arrays,
)
from zlc_runtime import DatasetCoverage
from zlc_runtime import DatasetOutputDeclaration, LiveDatasetOutput
from zlc_runtime import SignalValue
from zlc_plot import (
    IMAGE_POINT_OVERLAY_CONTRACT,
    IMAGE_POINT_OVERLAY_GEOMETRY_RECORD,
    image_point_overlay_geometry,
)

from zlc_atom.devices.camera.photoelectrons import PHOTOELECTRONS
from zlc_atom.nodes.calibration import ReadoutModelKind, TrapCalibration
from zlc_atom.nodes.calibration.calibration import classify_threshold
OCCUPANCY_OUTPUTS = (
    DatasetOutputDeclaration("counts", "occupancy.counts.v1"),
    DatasetOutputDeclaration("occupied", IMAGE_POINT_OVERLAY_CONTRACT),
    DatasetOutputDeclaration("frame_judged", "occupancy.frame_judged.v1"),
)


@dataclass(frozen=True)
class OccupancyResult:
    #: The typed snapshots are the result truth.  Array conveniences below
    #: are views of them, never a second stored copy.
    artifacts: Mapping[str, OwnedSnapshot]

    def __post_init__(self) -> None:
        artifacts = dict(self.artifacts)
        expected = {output.name for output in OCCUPANCY_OUTPUTS}
        if set(artifacts) != expected or any(
            not isinstance(value, OwnedSnapshot) for value in artifacts.values()
        ):
            raise ValueError("occupancy result must contain every typed output snapshot")
        object.__setattr__(self, "artifacts", MappingProxyType(artifacts))

    @property
    def counts(self) -> np.ndarray:
        return self.artifacts["counts"].block.values

    @property
    def occupied(self) -> np.ndarray:
        return self.artifacts["occupied"].block.values

    @property
    def frame_judged(self) -> np.ndarray:
        return self.artifacts["frame_judged"].block.values


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

        model_kind = self.model.kind
        self._runtime = None
        self.model = self.calibration.select_model(model_kind)
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
        self._refuse_a_different_unit(record)

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
            self.model = self._runtime.select_model(model_kind)
        elif tuple(binning) != tuple(contract.binning_yx):
            raise ValueError(
                f"camera binning {tuple(binning)} differs from calibration "
                f"{tuple(contract.binning_yx)}"
            )

    def _refuse_a_different_unit(self, record: Mapping[str, object]) -> None:
        """A threshold is a number of somethings; the somethings must match.

        Counts and photoelectrons differ by an affine map, so a run read in
        one and classified by thresholds fitted in the other is not a little
        wrong -- every site reads the same way.  Both sides record which they
        are, so the mismatch is refused rather than discovered in the data.
        """

        trained = self.calibration.report.get("run_record")
        if not isinstance(trained, Mapping):
            return
        wanted = bool((trained.get("request") or {}).get(PHOTOELECTRONS, False))
        got = bool((record.get("parameters") or {}).get(PHOTOELECTRONS, False))
        if wanted != got:
            names = {True: "photoelectrons", False: "counts"}
            raise ValueError(
                f"these frames are in {names[got]} and the calibration was "
                f"trained in {names[wanted]}; its thresholds do not apply"
            )

    @property
    def dataset_output_declarations(self) -> tuple[DatasetOutputDeclaration, ...]:
        return OCCUPANCY_OUTPUTS

    def signal_key(self, output_name: str) -> str:
        names = {declaration.name for declaration in OCCUPANCY_OUTPUTS}
        if str(output_name) not in names:
            raise KeyError(f"unknown occupancy output {output_name!r}")
        return f"@logic/{self.instance_id}/{output_name}"

    def _output_schemas(
        self,
        source: DatasetSchema,
    ) -> dict[str, DatasetSchema]:
        site_axis = self.readout.site_map.site_axis

        def with_cell(cell: ValueSchema) -> DatasetSchema:
            return DatasetSchema(
                source.repeat_axis,
                source.point_table,
                source.grid_topology,
                cell,
            )

        site_validity = ValidityContract.components(site_axis.axis_id)
        return {
            "counts": with_cell(
                ValueSchema(
                    (site_axis,),
                    site_validity,
                    np.dtype("<f8"),
                    source.cell_schema.value_unit,
                )
            ),
            "occupied": with_cell(
                ValueSchema((site_axis,), site_validity, np.dtype("?"), "1")
            ),
        }

    @staticmethod
    def _snapshot(
        source: OwnedSnapshot,
        schema: DatasetSchema,
        values: object,
        validity: object,
    ) -> OwnedSnapshot:
        return owned_snapshot_from_arrays(
            schema,
            values,
            source.block.revision,
            validity=validity,
            stream_generation=source.ref.stream_generation,
        )

    def process(self, frames: OwnedSnapshot) -> OccupancyResult:
        """Classify one source event snapshot without reconstructing history."""

        if not isinstance(frames, OwnedSnapshot):
            raise TypeError("occupancy process requires zlc_data.OwnedSnapshot")
        self._source_point_column(frames)
        images = np.asarray(frames.block.values)
        repeats, points = images.shape[:2]
        n_sites = self.readout.n_sites
        flat = images.reshape((repeats * points, *images.shape[2:]))
        source_validity = frames.expanded_validity()
        cell_valid = np.all(
            source_validity,
            axis=tuple(range(2, source_validity.ndim)),
        ).reshape(-1)
        counts = np.full((flat.shape[0], n_sites), np.nan, dtype="<f8")
        for index in np.flatnonzero(cell_valid):
            counts[index] = self.readout.signals(
                flat[index],
                model_kind=self.model.kind,
            )
        model = self.model
        site_usable = (
            self.readout.site_map.valid_sites
            & model.usable_sites
            & np.isfinite(model.thresholds)
        )
        valid = cell_valid[:, None] & site_usable[None, :]
        counts[~valid] = np.nan
        occupied = classify_threshold(counts, model.thresholds) & valid
        counts = counts.reshape((repeats, points, n_sites))
        occupied = occupied.reshape((repeats, points, n_sites))
        valid = valid.reshape((repeats, points, n_sites))
        schemas = self._output_schemas(frames.block.schema)
        artifacts = {
            "counts": self._snapshot(frames, schemas["counts"], counts, valid),
            "occupied": self._snapshot(
                frames, schemas["occupied"], occupied, valid
            ),
            # The source event already owns these immutable bytes, axes and
            # validity.  Runtime restamps the sibling under the processor's
            # route; copying it here would create a second frame truth.
            "frame_judged": frames,
        }
        return OccupancyResult(artifacts)

    def _live_outputs(
        self,
        result: OccupancyResult,
        *,
        source: SignalValue,
        frames_signal: str,
    ) -> dict[str, LiveDatasetOutput]:
        run_record = {
            "node": self.instance_id,
            IMAGE_POINT_OVERLAY_GEOMETRY_RECORD: image_point_overlay_geometry(
                source.snapshot,
                self.readout.site_map.centers_xy,
                self.readout.site_map.site_ids,
                status_axis=self.readout.site_map.site_axis,
                labels=tuple(
                    str(index)
                    for index in range(1, self.readout.site_map.n_sites + 1)
                ),
                coordinates_are_indices=True,
            ),
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
        event_schema = source.snapshot.block.schema
        event_cells = (
            event_schema.repeat_axis.size * event_schema.point_table.row_count
        )
        exact = isinstance(source.coverage, DatasetCoverage)
        if exact:
            if source.canonical_schema is None or source.cell_origin is None:
                raise ValueError("finite source event lacks canonical placement")
            canonical = self._output_schemas(source.canonical_schema)
            canonical["frame_judged"] = source.canonical_schema
            origin = source.cell_origin
        elif source.coverage is None:
            exact = True
            canonical = self._output_schemas(event_schema)
            canonical["frame_judged"] = event_schema
            origin = (0, 0)
        else:
            canonical = {}
            origin = None
        outputs: dict[str, LiveDatasetOutput] = {}
        for declaration in OCCUPANCY_OUTPUTS:
            snapshot = result.artifacts[declaration.name]
            coverage = (
                DatasetCoverage(event_cells, event_cells)
                if source.coverage is None
                else source.coverage
            )
            output_schema = canonical.get(declaration.name) if exact else None
            output_origin = origin if exact else None
            outputs[declaration.name] = LiveDatasetOutput(
                declaration,
                snapshot,
                coverage,
                run_record,
                output_schema,
                output_origin,
            )
        return outputs

    def evaluate(self, signal_value: SignalValue) -> dict[str, LiveDatasetOutput]:
        if not isinstance(signal_value, SignalValue):
            raise TypeError("occupancy evaluate requires zlc_runtime.SignalValue")
        snapshot = signal_value.snapshot
        # WHERE the calibration sits comes first.  Every check below is made
        # against the crop this run is taking, and that crop is a fact carried
        # by the run record -- so reading it is not a validation step, it is
        # what the validation is done against.  Checked in the other order,
        # the frame shape was compared with the crop the calibration was
        # MEASURED on and a run that had moved its ROI was refused before the
        # translation it needed had been computed.
        self._validate_source_run_record(signal_value)
        self._source_point_column(snapshot)
        result = self.process(snapshot)
        return self._live_outputs(
            result,
            source=signal_value,
            frames_signal=self.source_signal or signal_value.name,
        )

__all__ = [
    "OCCUPANCY_OUTPUTS",
    "OccupancyProcessor",
    "OccupancyResult",
]
