"""Minimal same-shot occupancy processor."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
from zlc_data import (
    CoordinateFrameId,
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
from zlc_data.value import dataset_validity_storage
from zlc_runtime import DatasetCoverage
from zlc_runtime import DatasetOutputDeclaration, LiveDatasetOutput
from zlc_runtime import SignalValue
from zlc_plot import (
    IMAGE_POINT_OVERLAY_CONTRACT,
    IMAGE_POINT_OVERLAY_GEOMETRY_RECORD,
    image_point_overlay_geometry,
)

from zlc_atom.nodes._framework.frames import REST, assign_frames, canonical_frames, check_claims, claim_names
from zlc_atom.nodes.calibration import ReadoutModelKind, TrapCalibration
from zlc_atom.nodes.calibration.calibration import classify_threshold, reads_photoelectrons


#: The frame a camera's pixel axes are counted in: the whole sensor's own.
_SENSOR_PIXELS = CoordinateFrameId("sensor_pixel_xy")


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


@dataclass(frozen=True)
class CalibrationReading:
    """One calibration and the frames of every cycle it reads.

    ``frames`` are claims (:mod:`zlc_atom.nodes._framework.frames`): frame
    numbers from 0, ``"first"``, ``"last"`` and ``"rest"``; none at all is a
    calibration kept but not read.  ``path`` is the file it was read from,
    for the run record and for naming it in a refusal.
    """

    calibration: TrapCalibration
    frames: tuple[int | str, ...] = (REST,)
    path: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "frames", canonical_frames(self.frames))
        if self.path is not None:
            object.__setattr__(self, "path", Path(self.path).expanduser().resolve())


class OccupancyProcessor:
    """Evaluate the calibrated readout once per camera frame.

    Every frame of a cycle reads the SAME sites, with the calibration that
    claims it: a load frame and a readout frame taken under different
    exposures are each read with the thresholds and kernels trained on frames
    like them.  Which calibration reads which frame is resolved against the
    frames that actually arrive, so "first", "last" and "rest" follow a cycle
    that gains or loses a frame.  A frame two calibrations read, or none
    does, refuses the run by name; a middle number the cycle does not carry
    between its ends reads nothing and refuses nothing.  One calibration on
    its own reads every frame.

    A calibration claiming no frame is PARKED: the operator keeps it at hand
    without using it.  Only its file goes on, into the run record; it is
    never read, and never checked against the calibrations that are -- the
    one kept aside is usually the one that would not fit.  ``parked`` names
    such files directly, for a caller that never decoded them.
    """

    def __init__(
        self,
        calibrations: TrapCalibration | Sequence[CalibrationReading],
        *,
        parked: Sequence[Path | None] = (),
        source_signal: str | None = None,
        model_kind: ReadoutModelKind | None = None,
    ) -> None:
        given = (
            (CalibrationReading(calibrations),)
            if isinstance(calibrations, TrapCalibration)
            else tuple(calibrations)
        )
        if any(not isinstance(reading, CalibrationReading) for reading in given):
            raise TypeError("occupancy calibrations must be CalibrationReading")
        readings = tuple(reading for reading in given if reading.frames)
        #: The files kept but not read, for the run record only.
        self.parked: tuple[Path | None, ...] = (
            *(reading.path for reading in given if not reading.frames),
            *(None if path is None else Path(path).expanduser().resolve() for path in parked),
        )
        if not readings and not self.parked:
            raise ValueError("occupancy needs at least one calibration")
        #: How a refusal names each calibration: as the editor's rows name
        #: it (claim_names, which reads the parked files too) when it has a file.
        filed = claim_names([{"path": "" if path is None else str(path)}
                             for path in (*(reading.path for reading in readings), *self.parked)])
        self._names = tuple(
            filed[index] if reading.path is not None
            else "the calibration" if len(readings) == 1
            else f"calibration {index + 1}"
            for index, reading in enumerate(readings)
        )
        check_claims(
            (*(reading.frames for reading in readings), *((),) * len(self.parked)),
            (*self._names, *filed[len(readings):]),
            what="calibration",
        )
        self.readings = readings
        paths = [reading.path for reading in readings]
        repeated = sorted({name for name, path in zip(self._names, paths)
                           if path is not None and paths.count(path) > 1})
        if repeated:
            raise ValueError(f"{repeated[0]} is listed twice: give it every frame it reads in one row")
        first = readings[0].calibration
        for name, reading in zip(self._names, readings):
            _require_calibration(reading.calibration, name)
            if tuple(reading.calibration.site_map.site_ids) != tuple(first.site_map.site_ids):
                raise ValueError(
                    f"{name} names different sites ({reading.calibration.n_sites}) from "
                    f"{self._names[0]} ({first.n_sites}): occupancy reads one set of "
                    "sites in every frame"
                )
        #: The model kind every frame reads with, resolved once.
        if model_kind is None:
            defaults = {
                name: reading.calibration.default_model_kind
                for name, reading in zip(self._names, readings)
            }
            if len(set(defaults.values())) > 1:
                raise ValueError(
                    "these calibrations default to different readout models ("
                    + ", ".join(f"{name}: {kind.value}" for name, kind in defaults.items())
                    + "); choose one Readout model for all of them"
                )
            self._model_kind = next(iter(defaults.values()))
        else:
            for name, reading in zip(self._names, readings):
                try:
                    reading.calibration.select_model(model_kind)
                except KeyError:
                    raise ValueError(
                        f"{name} has no {model_kind.value} readout model"
                    ) from None
            self._model_kind = first.select_model(model_kind).kind
        #: Every calibration as placed against the crop the RUN is taking,
        #: once the frames say what that crop is; until then each is read on
        #: the crop it was measured on.
        self.placed: tuple[TrapCalibration, ...] = tuple(
            reading.calibration for reading in readings
        )
        #: Which calibration reads each frame, per frame count seen.
        self._entries: dict[int, tuple[int, ...]] = {}
        self.source_signal = None if source_signal is None else str(source_signal).strip()

    @property
    def model_kind(self) -> ReadoutModelKind:
        """The readout model every frame reads with."""

        return self._model_kind

    def entries_for(self, count: int) -> tuple[int, ...]:
        """Which calibration (by index) reads each frame of a ``count``-frame cycle."""

        entries = self._entries.get(count)
        if entries is None:
            entries = assign_frames(
                tuple(reading.frames for reading in self.readings),
                count,
                self._names,
                what="calibration",
            )
            self._entries[count] = entries
        return entries

    def _validate_images(self, snapshot: OwnedSnapshot) -> None:
        """Validate image cells; Repeat/Point axes pass through unchanged."""

        schema = snapshot.block.schema
        axes = schema.cell_domain.axes
        if tuple(axis.role for axis in axes) != (SPATIAL_Y, SPATIAL_X):
            raise ValueError(
                "occupancy frames must declare exactly SPATIAL_Y, SPATIAL_X cell axes"
            )
        observed = tuple(int(axis.size) for axis in axes)
        for name, placed in zip(self._names, self.placed):
            expected = placed.frame_contract.image_shape
            if observed != expected:
                raise ValueError(
                    f"frame shape {observed} differs from the crop {name} "
                    f"is placed against {expected}"
                )

    def _place_against_the_frames(self, source: SignalValue) -> None:
        """Place every calibration on the crop these frames cover, in their unit.

        The FRAMES say both.  Their two spatial axes are the sensor pixels
        they cover -- the origin, the binning as the step, the size -- and
        their value unit says counts or photoelectrons.  Both travel with the
        frames through whatever carries them on: a scan of them, a derive of
        them.  The camera's run record does not: read from that record alone,
        a scan's frames were neither placed nor checked, and a scan whose ROI
        had moved read every site off by the move, or classified counts by
        thresholds trained in photoelectrons.  The record still says the one
        thing the frames cannot, the shape of the whole sensor.
        """

        self.placed = tuple(reading.calibration for reading in self.readings)
        schema = source.snapshot.block.schema
        self._refuse_a_different_unit(schema.value_schema.value_unit)
        snapshots = source.run_record.get("device_snapshots")
        camera = snapshots.get("camera") if isinstance(snapshots, Mapping) else None
        sensor = camera.get("sensor_shape_yx") if isinstance(camera, Mapping) else None
        if sensor is not None:
            sensor = tuple(int(item) for item in sensor)
            for name, placed in zip(self._names, self.placed):
                expected = placed.frame_contract.sensor_shape
                if expected is not None and sensor != expected:
                    raise ValueError(
                        f"camera sensor shape {sensor} differs from {name}'s {expected}"
                    )

        # A run may crop the sensor differently from the calibration: where a
        # trap IS, is a fact about the sensor, so a different ROI numbers the
        # same places differently and the calibration can be read against it.
        # Only a crop that does not COVER the sites is refused, and only by
        # the calibration itself -- it owns both crops, so it owns the
        # translation.  Binning is refused there too: it changes what a pixel
        # means, and with it every threshold measured in pixels.  Frames whose
        # axes name no sensor pixels are read on the crop each calibration
        # was measured on, which only a frame of that very shape passes.
        axes = schema.cell_domain.axes
        if len(axes) != 2 or any(axis.coordinate_frame != _SENSOR_PIXELS for axis in axes):
            return
        origin = tuple(int(axis.coordinate_at(0)) for axis in axes)
        binning = tuple(
            int(axis.coordinate_at(1)) - int(axis.coordinate_at(0))
            if axis.size > 1
            # One pixel has no step to say otherwise.
            else int(self.placed[0].frame_contract.binning_yx[index])
            for index, axis in enumerate(axes)
        )
        image_shape = tuple(int(axis.size) for axis in axes)
        roi = (
            origin[1], origin[0],
            image_shape[1] * binning[1], image_shape[0] * binning[0],
        )
        # Every calibration is read against the same crop.
        self.placed = tuple(
            placed.rebased(roi, binning, image_shape) for placed in self.placed
        )

    def _refuse_a_different_unit(self, value_unit: str | None) -> None:
        """A threshold is a number of somethings; the somethings must match.

        Counts and photoelectrons differ by an affine map, so a run read in
        one and classified by thresholds fitted in the other is not a little
        wrong -- every site reads the same way.  Both sides say which they
        are -- a frame in its unit, the camera's count unit or a plain number
        (a photoelectron is one: no unit, or "1" once a derive has done
        arithmetic on it) -- so the mismatch is refused rather than
        discovered in the data.
        """

        got = value_unit in (None, "1")
        names = {True: "photoelectrons", False: "counts"}
        # Every calibration a frame is read with: each frame is classified by
        # the thresholds of the one that claims it.
        for name, reading in zip(self._names, self.readings):
            wanted = reads_photoelectrons(reading.calibration)
            if wanted is not None and wanted != got:
                raise ValueError(
                    f"these frames are in {names[got]} and {name} was "
                    f"trained in {names[wanted]}; its thresholds do not apply"
                )

    @property
    def dataset_output_declarations(self) -> tuple[DatasetOutputDeclaration, ...]:
        return OCCUPANCY_OUTPUTS

    def _output_schemas(
        self,
        source: DatasetSchema,
    ) -> dict[str, DatasetSchema]:
        # Every calibration names the same sites, so any one names the axis.
        site_axis = self.placed[0].site_map.site_axis

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
        n_sites = self.placed[0].n_sites
        flat = images.reshape((repeats * points, *images.shape[2:]))
        # Which frame of its cycle each cell is, and so which calibration
        # reads it.  The Point axis with the readout-event role counts a
        # cycle's frames, whatever else the Point domain carries beside it
        # (a scan's own axes, say).
        frame_of_cell, count = self._frame_positions(frames.block.schema, repeats, points)
        entry_of_cell = np.asarray(self.entries_for(count))[frame_of_cell]
        # Judged on the axes validity may vary on -- a whole frame, unless the
        # source names components -- not broadcast over every pixel first.
        judged = dataset_validity_storage(frames.block.validity, frames.block.schema)
        cell_valid = np.all(judged, axis=tuple(range(2, judged.ndim))).reshape(-1)
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
            entry_of_valid = entry_of_cell[valid_cells]
            for entry, readout in enumerate(self.placed):
                selected = valid_cells[entry_of_valid == entry]
                if selected.size:
                    counts[selected] = readout.signals_of_frames(
                        flat,
                        model_kind=self._model_kind,
                        frame_indices=None if selected.size == flat.shape[0] else selected,
                    )
        # Every frame judges its sites by the thresholds and the usable set
        # of the calibration IT reads with: one row of each per cell.
        site_usable, thresholds = self._verdict_tables(entry_of_cell)
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

    @staticmethod
    def _frame_axis(schema: DatasetSchema):
        return next(
            (axis for axis in schema.point_domain.axes if axis.role == READOUT_EVENT),
            None,
        )

    def _frame_positions(
        self, schema: DatasetSchema, repeats: int, points: int
    ) -> tuple[np.ndarray, int]:
        """The frame (from 0) of every cell of the flattened event, and the cycle's frame count.

        A source without a frame axis is one frame per cycle, frame 0.
        """

        axis = self._frame_axis(schema)
        if axis is None:
            return np.zeros(repeats * points, dtype=int), 1
        # Read off the domain's own codes: a Point domain is a MAPPED one, not
        # a dense grid in declaration order.  A scan declares the frame axis
        # first and lays it innermost, and names a repeated value once, so
        # unravelling the axis sizes numbered a scan's frames 0,0,1,1,2,2 --
        # or ran out of bounds.
        codes = np.asarray(schema.point_domain.codes(axis.axis_id))
        return np.tile(codes, repeats), int(axis.size)

    def _verdict_tables(self, entry_of_cell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per cell: the usable sites and the thresholds of the calibration reading it."""

        usable, thresholds = [], []
        for readout in self.placed:
            model = readout.select_model(self._model_kind)
            usable.append(
                readout.site_map.valid_sites & model.usable_sites & np.isfinite(model.thresholds)
            )
            thresholds.append(np.asarray(model.thresholds, dtype=float))
        return np.stack(usable)[entry_of_cell], np.stack(thresholds)[entry_of_cell]

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
        axis = self._frame_axis(source.canonical_schema or source.snapshot.block.schema)
        entries = self.entries_for(1 if axis is None else int(axis.size))
        # The overlay draws the sites where the calibration reading the
        # first frame places them.
        overlay = self.placed[entries[0]].site_map
        return {
            IMAGE_POINT_OVERLAY_GEOMETRY_RECORD: image_point_overlay_geometry(
                source.snapshot,
                overlay.centers_xy,
                overlay.site_ids,
                status_axis=overlay.site_axis,
                labels=tuple(str(index) for index in range(1, overlay.n_sites + 1)),
                coordinates_are_indices=True,
            ),
            "parameters": {
                "frames_signal": self.source_signal or source.name,
                # What was asked for beside what it came to: the claims
                # follow the frame count, so the record says both.  A
                # parked file is recorded as kept, reading nothing.
                "calibrations": [
                    {
                        "path": None if reading.path is None else str(reading.path),
                        "frames": list(reading.frames),
                        "read_frames": [
                            frame for frame, entry in enumerate(entries) if entry == index
                        ],
                    }
                    for index, reading in enumerate(self.readings)
                ] + [
                    {"path": None if path is None else str(path), "frames": [], "read_frames": []}
                    for path in self.parked
                ],
                "model_kind": self._model_kind.value,
            },
        }

    def evaluate(self, signal_value: SignalValue) -> dict[str, LiveDatasetOutput]:
        if not isinstance(signal_value, SignalValue):
            raise TypeError("occupancy evaluate requires zlc_runtime.SignalValue")
        snapshot = signal_value.snapshot
        # WHERE the calibration sits comes first.  Every check below is made
        # against the crop this run is taking, and that crop is a fact carried
        # by the frames' own axes -- so reading it is not a validation step, it
        # is what the validation is done against.  Checked in the other order,
        # the frame shape was compared with the crop the calibration was
        # MEASURED on and a run that had moved its ROI was refused before the
        # translation it needed had been computed.
        self._place_against_the_frames(signal_value)
        result = self.process(snapshot)
        return self._live_outputs(
            result,
            source=signal_value,
        )

__all__ = [
    "OCCUPANCY_OUTPUTS",
    "CalibrationReading",
    "OccupancyProcessor",
    "OccupancyResult",
]
