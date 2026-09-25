"""Minimal same-shot occupancy processor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
from zlc_data import (
    DatasetSchema,
    DomainSpec,
    OwnedSnapshot,
    READOUT_EVENT,
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
from zlc_atom.nodes.calibration import ReadoutModel, ReadoutModelKind, TrapCalibration
from zlc_atom.nodes.calibration.calibration import classify_threshold


def _require_calibration(candidate: object, what: str) -> None:
    if not isinstance(candidate, TrapCalibration):
        raise TypeError(f"{what} must be TrapCalibration")
    if candidate.site_map.coordinate_frame != "image_pixel_xy":
        raise ValueError(
            f"occupancy requires {what} centers in image_pixel_xy coordinates"
        )


OCCUPANCY_OUTPUTS = (
    DatasetOutputDeclaration("counts", "occupancy.counts", index_by_source=True),
    DatasetOutputDeclaration(
        "occupied", IMAGE_POINT_OVERLAY_CONTRACT, index_by_source=True
    ),
    DatasetOutputDeclaration("frame_judged", "occupancy.frame_judged"),
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
    """Evaluate the calibrated readout once per camera frame.

    Every frame of a cycle reads the SAME sites; a frame may read them with
    a calibration of its own -- a load frame and a readout frame taken under
    different exposures, each with the thresholds and kernels trained on
    frames like it -- and every frame not given one reads the shared one.
    Frames are numbered from 1, the way the operator sees them.
    """

    def __init__(
        self,
        calibration: TrapCalibration,
        *,
        calibration_by_frame: Mapping[int, TrapCalibration] | None = None,
        calibration_path: str | Path | None = None,
        calibration_paths_by_frame: Mapping[int, str | Path] | None = None,
        producer: str = "occupancy",
        source_signal: str | None = None,
        model_kind: ReadoutModelKind | None = None,
    ) -> None:
        _require_calibration(calibration, "calibration")
        self.calibration = calibration
        #: The model kind every frame reads with, resolved once.
        self._model_kind = calibration.select_model(model_kind).kind
        by_frame: dict[int, TrapCalibration] = {}
        for frame, candidate in dict(calibration_by_frame or {}).items():
            number = int(frame)
            if number < 1:
                raise ValueError("calibration frames are counted from 1")
            _require_calibration(candidate, f"frame {number} calibration")
            if tuple(candidate.site_map.site_ids) != tuple(calibration.site_map.site_ids):
                raise ValueError(
                    f"frame {number}'s calibration names different sites "
                    f"({candidate.n_sites}) from the shared one ({calibration.n_sites}): "
                    "occupancy reads one set of sites in every frame"
                )
            try:
                candidate.select_model(self._model_kind)
            except KeyError:
                raise ValueError(
                    f"frame {number}'s calibration has no {self._model_kind.value} readout model"
                ) from None
            by_frame[number] = candidate
        self.calibration_by_frame: Mapping[int, TrapCalibration] = MappingProxyType(by_frame)
        #: Every calibration as placed against the crop the RUN is taking,
        #: once a run record says what that crop is; key 0 is the shared one.
        #: Until then each is read on the crop it was measured on.
        self._placed: dict[int, TrapCalibration] = {0: calibration, **by_frame}
        self.calibration_path = (
            None
            if calibration_path is None
            else Path(calibration_path).expanduser().resolve()
        )
        self.calibration_paths_by_frame: Mapping[int, Path] = MappingProxyType({
            int(frame): Path(path).expanduser().resolve()
            for frame, path in dict(calibration_paths_by_frame or {}).items()
        })
        self.instance_id = str(producer).strip()
        if not self.instance_id:
            raise ValueError("producer must be non-empty")
        self.source_signal = None if source_signal is None else str(source_signal).strip()

    def readout_for(self, frame: int) -> TrapCalibration:
        """The calibration frame ``frame`` (from 1) reads with; the shared one when it names none."""

        return self._placed.get(frame, self._placed[0])

    @property
    def model(self) -> ReadoutModel:
        """The shared calibration's model of the kind every frame reads with."""

        return self.readout.select_model(self._model_kind)

    def _validate_images(self, snapshot: OwnedSnapshot) -> None:
        """Validate image cells; Repeat/Point axes pass through unchanged."""

        schema = snapshot.block.schema
        axes = schema.cell_domain.axes
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

    @property
    def readout(self) -> TrapCalibration:
        """The shared calibration as it applies to the frames actually arriving."""

        return self._placed[0]

    def _validate_source_run_record(self, source: SignalValue) -> None:
        """Check only structural camera facts present on the parent."""

        self._placed = {0: self.calibration, **self.calibration_by_frame}
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
        if sensor is not None:
            for frame, placed in self._placed.items():
                expected = placed.frame_contract.sensor_shape
                if expected is not None and sensor != expected:
                    which = "calibration" if frame == 0 else f"frame {frame}'s calibration"
                    raise ValueError(
                        f"camera sensor shape {sensor} differs from {which} {expected}"
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
            image_shape = (
                int(shape[0]) // int(binning[0]),
                int(shape[1]) // int(binning[1]),
            )
            # Every calibration is read against the same crop.
            self._placed = {
                frame: placed.rebased(roi, binning, image_shape)
                for frame, placed in self._placed.items()
            }
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

    def _output_schemas(
        self,
        source: DatasetSchema,
    ) -> dict[str, DatasetSchema]:
        site_axis = self.readout.site_map.site_axis

        site_domain = DomainSpec((site_axis.size,), (site_axis,))

        def with_value(value: ValueSchema) -> DatasetSchema:
            return DatasetSchema(
                source.repeat_domain,
                source.point_domain,
                site_domain,
                value,
            )

        site_validity = ValidityContract.components(site_axis.axis_id)
        return {
            "counts": with_value(
                ValueSchema(
                    site_validity,
                    np.dtype("<f4"),
                    source.value_schema.value_unit,
                    name="counts",
                )
            ),
            "occupied": with_value(
                ValueSchema(site_validity, np.dtype("?"), "1", name="occupied")
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
        frames = frames.materialize()
        self._validate_images(frames)
        images = np.asarray(frames.block.values)
        repeats, points = images.shape[:2]
        n_sites = self.readout.n_sites
        flat = images.reshape((repeats * points, *images.shape[2:]))
        # Which frame of its cycle each cell is.  The Point axis with the
        # readout-event role counts a cycle's frames, whatever else the
        # Point domain carries beside it (a scan's own axes, say).
        frame_of_cell = self._frame_numbers(frames.block.schema, repeats, points)
        source_validity = frames.expanded_validity()
        cell_valid = np.all(
            source_validity,
            axis=tuple(range(2, source_validity.ndim)),
        ).reshape(-1)
        # Single precision IS this measurement: a site's signal is a sum of
        # sensor counts, and 24 bits of mantissa resolve every one of them
        # exactly up to sixteen million.  It is also the number the verdict
        # below is read from, so the published count and the published
        # verdict cannot be derived from two different values of the same
        # measurement.
        counts = np.full((flat.shape[0], n_sites), np.nan, dtype="<f4")
        # A signal single precision cannot hold overflows in this cast.  It
        # is judged below, where every site whose count is not a finite
        # number is marked invalid, so the cast need not also warn.
        with np.errstate(over="ignore"):
            # One read per calibration over every cell it reads, not one
            # per cell: a 42-frame cycle is 42 windows gathered at once.
            valid_cells = np.flatnonzero(cell_valid)
            own_frames = tuple(self.calibration_by_frame)
            frame_of_valid = frame_of_cell[valid_cells]
            for frame, readout in self._placed.items():
                selected = valid_cells[
                    ~np.isin(frame_of_valid, own_frames) if frame == 0 else frame_of_valid == frame
                ]
                if selected.size:
                    counts[selected] = readout.signals_of_frames(
                        flat[selected], model_kind=self._model_kind,
                    )
        # Every frame judges its sites by the thresholds and the usable set
        # of the calibration IT reads with: one row of each per cell.
        site_usable, thresholds = self._verdict_tables(frame_of_cell)
        # A verdict needs a number to read.  Where the readout produced none
        # -- a frame with NaN in the box, a sum beyond single precision --
        # there is no count and no verdict, and the site is INVALID for this
        # cell: publishing it as a valid EMPTY would let a survival or
        # agreement panel count an unjudgeable trial as an atom lost.
        valid = cell_valid[:, None] & site_usable & np.isfinite(counts)
        counts[~valid] = np.nan
        occupied = classify_threshold(counts, thresholds) & valid
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

    def _frame_numbers(self, schema: DatasetSchema, repeats: int, points: int) -> np.ndarray:
        """The frame number (from 1) of every cell of the flattened event.

        Zero everywhere when the source has no frame axis: then only the
        shared calibration applies.
        """

        axes = schema.point_domain.axes
        position = next(
            (index for index, axis in enumerate(axes) if axis.role == READOUT_EVENT),
            None,
        )
        if position is None:
            if self.calibration_by_frame:
                raise ValueError(
                    "these frames have no frame axis to read per-frame calibrations against"
                )
            return np.zeros(repeats * points, dtype=int)
        sizes = tuple(int(axis.size) for axis in axes)
        beyond = sorted(frame for frame in self.calibration_by_frame if frame > sizes[position])
        if beyond:
            raise ValueError(
                f"frame {beyond[0]} has its own calibration but a cycle has only "
                f"{sizes[position]} frame(s)"
            )
        point_index = np.arange(repeats * points) % points
        return np.asarray(np.unravel_index(point_index, sizes)[position]) + 1

    def _verdict_tables(self, frame_of_cell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per cell: the usable sites and the thresholds of the calibration its frame reads with."""

        frames = (0, *self.calibration_by_frame)
        usable, thresholds = [], []
        for frame in frames:
            readout = self.readout_for(frame)
            model = readout.select_model(self._model_kind)
            usable.append(
                readout.site_map.valid_sites & model.usable_sites & np.isfinite(model.thresholds)
            )
            thresholds.append(np.asarray(model.thresholds, dtype=float))
        row = np.zeros(frame_of_cell.shape[0], dtype=int)
        for position, frame in enumerate(frames[1:], start=1):
            row[frame_of_cell == frame] = position
        return np.stack(usable)[row], np.stack(thresholds)[row]

    def _live_outputs(
        self,
        result: OccupancyResult,
        *,
        source: SignalValue,
    ) -> dict[str, LiveDatasetOutput]:
        event_schema = source.snapshot.block.schema
        event_cells = (
            event_schema.repeat_domain.size * event_schema.point_domain.size
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
                output_schema,
                output_origin,
            )
        return outputs

    def describe_run(self, inputs: Mapping[str, SignalValue]) -> dict[str, object]:
        source = next(iter(inputs.values()))
        return {
            "node": self.instance_id,
            IMAGE_POINT_OVERLAY_GEOMETRY_RECORD: image_point_overlay_geometry(
                source.snapshot,
                self.readout.site_map.centers_xy,
                self.readout.site_map.site_ids,
                status_axis=self.readout.site_map.site_axis,
                labels=tuple(str(index) for index in range(1, self.readout.site_map.n_sites + 1)),
                coordinates_are_indices=True,
            ),
            "parameters": {
                "frames_signal": self.source_signal or source.name,
                "calibration_path": None if self.calibration_path is None else str(self.calibration_path),
                # Only when a frame read with its own: the record says what
                # the run used, and a run without one is recorded as before.
                **(
                    {
                        "calibration_paths_by_frame": {
                            str(frame): str(path)
                            for frame, path in sorted(self.calibration_paths_by_frame.items())
                        }
                    }
                    if self.calibration_paths_by_frame
                    else {}
                ),
                "model_kind": self.model.kind.value,
            },
        }

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
        result = self.process(snapshot)
        return self._live_outputs(
            result,
            source=signal_value,
        )

__all__ = [
    "OCCUPANCY_OUTPUTS",
    "OccupancyProcessor",
    "OccupancyResult",
]
