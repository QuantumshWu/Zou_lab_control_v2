"""Minimal same-shot occupancy processor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from zlc_data import OwnedSnapshot, READOUT_EVENT, SITE
from zlc_runtime import DatasetCoverage
from zlc_runtime import DatasetOutputDeclaration, LiveDatasetOutput
from zlc_runtime import (
    DerivedSignalOutput,
    SignalPublication,
    SignalValue,
)

from zlc_atom.devices.camera.contract import CameraFrameRecord
from zlc_atom.data import snapshot_from_array
from zlc_atom.nodes.calibration import TrapCalibration
from zlc_atom.nodes.calibration.calibration import classify_threshold
from zlc_atom.nodes._framework.descriptor import NodeKind, runtime_kind


_OUTPUT_DECLARATIONS = (
    DatasetOutputDeclaration("counts", "occupancy.counts.v1"),
    DatasetOutputDeclaration("occupied", "occupancy.occupied.v1"),
    DatasetOutputDeclaration("rate", "occupancy.rate.v1"),
)


@dataclass(frozen=True)
class OccupancyResult:
    counts: np.ndarray
    occupied: np.ndarray
    rate: np.ndarray
    source_publication: SignalPublication | None = None
    artifacts: dict[str, OwnedSnapshot] = field(default_factory=dict)

    def __post_init__(self) -> None:
        counts = np.asarray(self.counts, dtype="<f8")
        occupied = np.asarray(self.occupied, dtype=bool)
        rate = np.asarray(self.rate, dtype="<f8")
        if counts.shape != occupied.shape or counts.ndim < 1 or rate.shape != counts.shape[:-1]:
            raise ValueError("occupancy outputs must share leading dimensions")
        for value in (counts, occupied, rate):
            value.setflags(write=False)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "occupied", occupied)
        object.__setattr__(self, "rate", rate)
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


class OccupancyProcessor:
    #: Derived from the domain layer, never declared twice.
    kind = runtime_kind(NodeKind.PROCESSOR)

    """Evaluate the calibrated readout once per camera frame."""

    def __init__(self, calibration: TrapCalibration, *, signal_plane: object | None = None, producer: str = "occupancy", source_signal: str | None = None) -> None:
        if not isinstance(calibration, TrapCalibration):
            raise TypeError("calibration must be TrapCalibration")
        self.calibration = calibration
        self.signal_plane = signal_plane
        self.instance_id = str(producer).strip()
        if not self.instance_id:
            raise ValueError("producer must be non-empty")
        self.producer = self.instance_id
        self.source_signal = None if source_signal is None else str(source_signal).strip()

    @property
    def dataset_output_declarations(self) -> tuple[DatasetOutputDeclaration, ...]:
        return _OUTPUT_DECLARATIONS

    def signal_key(self, output_name: str) -> str:
        names = {declaration.name for declaration in _OUTPUT_DECLARATIONS}
        if str(output_name) not in names:
            raise KeyError(f"unknown occupancy output {output_name!r}")
        return f"@logic/{self.instance_id}/{output_name}"

    def _source_value(self, publication: SignalPublication) -> SignalValue:
        selected = self.source_signal
        if selected is not None:
            value = publication.value(selected)
            if value is None:
                raise ValueError(f"camera publication has no {selected!r} signal")
            return value
        candidates = [
            value
            for value in publication.signals.values()
            if value.name.rsplit("/", 1)[-1] == "frames"
        ]
        if len(candidates) != 1:
            raise ValueError("occupancy requires exactly one frames signal")
        self.source_signal = candidates[0].name
        return candidates[0]

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
            [self.calibration.signals(image) for image in images], dtype="<f8"
        )
        occupied = classify_threshold(counts, self.calibration.thresholds)
        counts = counts.reshape((*leading, self.calibration.n_sites))
        occupied = occupied.reshape((*leading, self.calibration.n_sites))
        rate = np.mean(occupied, axis=-1, dtype=float)
        site_roles = (READOUT_EVENT,) * max(counts.ndim - 2, 0) + (SITE,)
        rate_roles = (READOUT_EVENT,) * max(rate.ndim - 1, 0)
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
            "rate": snapshot_from_array(
                rate,
                producer=self.producer,
                signal="rate",
                roles=rate_roles,
                generation=generation,
                revision=revision,
            ),
        }
        return OccupancyResult(counts, occupied, rate, artifacts=artifacts)

    def _live_outputs(self, result: OccupancyResult) -> dict[str, LiveDatasetOutput]:
        outputs: dict[str, LiveDatasetOutput] = {}
        for declaration in _OUTPUT_DECLARATIONS:
            snapshot = result.artifacts[declaration.name]
            total = snapshot.block.schema.repeat_axis.size * snapshot.block.schema.point_table.row_count
            outputs[declaration.name] = LiveDatasetOutput(
                declaration,
                snapshot,
                DatasetCoverage(total, total),
            )
        return outputs

    def evaluate(self, signal_value: SignalValue) -> dict[str, LiveDatasetOutput]:
        if not isinstance(signal_value, SignalValue):
            raise TypeError("occupancy evaluate requires zlc_runtime.SignalValue")
        result = self.process(signal_value, **inherited_stamps(signal_value.snapshot))
        return self._live_outputs(result)

    def process_publication(self, publication: SignalPublication) -> OccupancyResult:
        if not isinstance(publication, SignalPublication):
            raise TypeError("publication must be SignalPublication")
        source = self._source_value(publication)
        result = self.process(source, **inherited_stamps(source.snapshot))
        result = OccupancyResult(result.counts, result.occupied, result.rate, publication, result.artifacts)
        if self.signal_plane is not None:
            bind = getattr(self.signal_plane, "bind_continuous_derived", None)
            publish = getattr(self.signal_plane, "publish_continuous_derived", None)
            if not callable(bind) or not callable(publish):
                raise TypeError("signal_plane must implement the frozen derived-publication contract")
            generation = bind(
                self.instance_id,
                source_name=source.name,
                expected_source_generation=publication.event_ref.generation,
                output_names=tuple(self.signal_key(declaration.name) for declaration in _OUTPUT_DECLARATIONS),
            )
            values = {
                self.signal_key(declaration.name): DerivedSignalOutput(result.artifacts[declaration.name])
                for declaration in _OUTPUT_DECLARATIONS
            }
            if not publish(self.instance_id, generation, (publication,), values):
                raise RuntimeError("signal plane rejected the occupancy lineage publication")
        return result


__all__ = ["OccupancyProcessor", "OccupancyResult"]
