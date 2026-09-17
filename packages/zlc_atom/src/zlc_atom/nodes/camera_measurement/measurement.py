"""Finite exact and live-monitor camera capture shared by the node."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from time import monotonic

import numpy as np
from zlc_data import (
    AxisId,
    AxisRoleId,
    AxisSpec,
    CoordinateFrameId,
    DomainSpec,
    OwnedSnapshot,
    READOUT_EVENT,
    SPATIAL_X,
    SPATIAL_Y,
)
from zlc_runtime import DatasetCoverage, MonitorCoverage
from zlc_runtime import (
    DatasetOutputDeclaration,
    LiveDatasetOutput,
    SignalValue,
)
from zlc_runtime import SignalPublication

from zlc_atom.devices.camera.contract import (
    CameraAdapter,
    CameraCaptureTerminalRecord,
    CameraFrameRecord,
    CameraWorkingPoint,
)
from zlc_atom.devices.camera.photoelectrons import PHOTOELECTRONS
from zlc_atom.data import cell_axis_id, snapshot_from_array


_CAMERA_FRAME_CONTRACT = "camera.frames"

#: The camera's ONE output: a cycle of frames on the dataset's POINT axis.
#: A cycle is one acquisition event; publishing its frames as N sibling
#: signals leaked the acquisition configuration into the signal vocabulary
#: (a panel bound to "frame_1" broke the moment frames_per_cycle changed),
#: and no consumer could see the whole cycle at once.  The frames live on
#: the POINT axis -- (repeat cycles) x (frame points) x (y, x) -- because a
#: frame is a point of the acquisition, not structure inside a pixel plane:
#: that is what lets a grid facet the frames side by side and lets a scan
#: compose them with scan axes in the same Point domain. The frame axis role
#: is READOUT_EVENT.
CAMERA_FRAMES_OUTPUT = DatasetOutputDeclaration("frames", _CAMERA_FRAME_CONTRACT)

#: How often a capture comes back to see whether it has been asked to stop.
#:
#: Cancellation is only ever seen BETWEEN reads, so the read length IS the
#: cancel latency, and it belongs to the LOOP rather than to the device.  It
#: used to be the CAMERA's timeout, which is a device fact about how long a
#: frame may take (2 s virtual, 10 s on the qCMOS), so an external-trigger
#: camera whose triggers had stopped sat inside one read until that deadline
#: -- and Stop, a Start queued behind the camera, and closing the console all
#: waited for it.  Fifty milliseconds is also the qCMOS driver's own wait
#: slice, so its SDK call cadence is unchanged.
#:
#: A monitor read is a POLL and takes whatever has arrived; a finite read
#: waits for a COMPLETE cycle and so must keep the camera's own timeout as
#: its deadline -- but it waits in slices of this, which is the difference
#: between a Stop that lands now and one that lands ten seconds from now.
_CANCEL_RESPONSE_SECONDS = 0.05
_RECEIVE_BUFFER_MIB = 128


def _frame_point_axis(producer: str, frames: int) -> AxisSpec:
    """The frame axis one cycle publishes in its Point domain."""

    return AxisSpec(
        AxisId(f"{producer}.frames.frame"),
        "frame",
        READOUT_EVENT,
        int(frames),
        tuple(range(int(frames))),
    )


@lru_cache(maxsize=32)
def _sensor_pixel_axis(
    producer: str,
    index: int,
    role: AxisRoleId,
    origin: int,
    step: int,
    size: int,
) -> AxisSpec:
    """One spatial axis of a frame, in the sensor's own pixels.

    Which pixels a frame covers is a fact about the working point, not about
    the frame -- a run publishes thousands of frames from one crop -- so the
    coordinates are built once per crop rather than per publication.

    The identity is the dataset's own, from the one function that generates
    it: saved boards, semantic choices and every downstream reference name a
    data axis by its id, so a producer adding coordinates must hand back the
    same name.
    """

    return AxisSpec(
        cell_axis_id(producer, CAMERA_FRAMES_OUTPUT.name, index, role),
        role.value,
        role,
        size,
        coordinates=tuple(origin + step * offset for offset in range(size)),
        unit="pixel",
        coordinate_frame=CoordinateFrameId("sensor_pixel_xy"),
    )


def _pixel_axes(
    point: "CameraWorkingPoint | None",
    shape_yx: tuple[int, int],
    *,
    producer: str,
) -> tuple[AxisSpec | AxisRoleId, AxisSpec | AxisRoleId]:
    """The sensor pixels a frame covers, as its two spatial axes.

    A frame is a crop of a sensor, and the numbers that mean anything about it
    are the sensor's own pixel coordinates: where the traps are, where an ROI
    starts, what a region drawn on the picture refers to.  Published without
    axes the picture was indexed from zero instead, so every coordinate on it
    was an offset into that particular crop -- the same region meant somewhere
    else the moment the ROI moved, which is exactly what a region does when it
    is used to set the ROI.

    The frame axis belongs to Point, so the cell is (y, x): those are the two
    cell axes the dataset will generate, in that order.
    """

    if point is None:
        return (SPATIAL_Y, SPATIAL_X)
    origin_y, origin_x = (int(value) for value in point.roi_origin_yx)
    step_y, step_x = (int(value) for value in getattr(point, "binning_yx", (1, 1)))
    height, width = (int(value) for value in shape_yx)
    return (
        _sensor_pixel_axis(
            producer, 0, SPATIAL_Y, origin_y, max(1, step_y), height
        ),
        _sensor_pixel_axis(
            producer, 1, SPATIAL_X, origin_x, max(1, step_x), width
        ),
    )


def frames_snapshot(
    cycles: "Sequence[Sequence[CameraFrameRecord]]",
    *,
    producer: str,
    generation: object,
    revision: int,
    working_point: "CameraWorkingPoint | None" = None,
    value_unit: str | None,
):
    """Cycles of frames as one dataset: (cycle) x (frame) x (y, x).

    Every publisher of camera frames -- the finite capture, the monitor slot,
    and a scan that owns its camera -- means exactly this dataset, so they
    build it here.  Three copies of the same stacking is how the frame point
    column drifts between the live view and the saved run.
    """

    frames = tuple(tuple(cycle) for cycle in cycles)
    if not frames:
        raise ValueError("camera publication requires at least one cycle")
    sizes = {len(cycle) for cycle in frames}
    if len(sizes) != 1 or not sizes.pop():
        raise ValueError("every published camera cycle must have the same frames")
    return snapshot_from_array(
        np.stack([np.asarray(record.image) for cycle in frames for record in cycle], axis=0)
        .reshape(len(frames), len(frames[0]), *np.asarray(frames[0][0].image).shape),
        producer=producer,
        signal=CAMERA_FRAMES_OUTPUT.name,
        point_axes=(_frame_point_axis(producer, len(frames[0])),),
        cell_axes=_pixel_axes(
            working_point,
            np.asarray(frames[0][0].image).shape,
            producer=producer,
        ),
        value_unit=value_unit,
        generation=str(getattr(generation, "value", generation)),
        revision=int(revision),
    )


def _finite_cycle_output(
    node: "CameraMeasurementNode",
    cycle: Sequence[CameraFrameRecord],
    index: int,
) -> LiveDatasetOutput:
    """One new cycle placed in the fixed authored finite run geometry."""

    event = frames_snapshot(
        (cycle,),
        producer=node.instance_id,
        generation=node.generation,
        revision=index + 1,
        working_point=node.actual_working_point,
        value_unit=node.frame_value_unit,
    )
    event_schema = event.block.schema
    (repeat_axis,) = event_schema.repeat_domain.axes
    canonical = replace(
        event_schema,
        repeat_domain=DomainSpec(
            (node.repeat,),
            (replace(repeat_axis, size=node.repeat),),
            (tuple(range(node.repeat)),),
        ),
    )
    frames = node.frames_per_cycle
    return LiveDatasetOutput(
        CAMERA_FRAMES_OUTPUT,
        event,
        DatasetCoverage((index + 1) * frames, node.repeat * frames),
        canonical_schema=canonical,
        cell_origin=(index, 0),
        event_record=node._camera_event_record(cycle),
    )


def _monitor_cycle_output(
    node: "CameraMeasurementNode",
    cycle: Sequence[CameraFrameRecord],
    revision: int,
) -> LiveDatasetOutput:
    event = frames_snapshot(
        (cycle,),
        producer=node.instance_id,
        generation=node.generation,
        revision=revision,
        working_point=node.actual_working_point,
        value_unit=node.frame_value_unit,
    )
    frames = node.frames_per_cycle
    return LiveDatasetOutput(
        CAMERA_FRAMES_OUTPUT,
        event,
        MonitorCoverage(frames, frames),
        event_record=node._camera_event_record(cycle),
    )


def _strict_cycle_ordinals(
    records: Sequence[CameraFrameRecord],
    *,
    expected_start: int,
    frames_per_cycle: int,
) -> tuple[CameraFrameRecord, ...]:
    cycle = tuple(records)
    expected = tuple(range(expected_start, expected_start + frames_per_cycle))
    observed = tuple(int(record.source_ordinal) for record in cycle)
    if len(cycle) != frames_per_cycle or observed != expected:
        raise RuntimeError(
            "camera cycle source ordinals are not contiguous: "
            f"expected {expected}, received {observed}"
        )
    return cycle


def _strict_terminal(
    terminal: CameraCaptureTerminalRecord,
    *,
    expected_frames: int,
    stopped: bool = False,
) -> CameraCaptureTerminalRecord:
    """The device's terminal, checked against the cycles this capture kept.

    The device must have stopped and joined, and it must have
    produced every frame the completed cycles account for.  A capture that was
    asked to stop may have left a partial cycle behind: the frames of the
    cycle it walked away from are the device's honest count, not a lie about
    the cycles it kept, so a surplus is accepted only for a stop.  A run that
    ended on its own must account for every frame exactly. Accepted FIFO data
    may still remain for the caller's ordinary-Stop drain.
    """

    if not (terminal.source_stopped and terminal.joined):
        raise RuntimeError(
            "camera terminal evidence is incomplete: the capture did not stop, "
            "drain and join"
        )
    produced = terminal.produced_count
    if produced < expected_frames or (produced != expected_frames and not stopped):
        raise RuntimeError(
            "camera terminal count differs from completed cycles: "
            f"completed cycles account for {expected_frames} frame(s), camera "
            f"produced {produced} (a partial cycle may be present)"
        )
    return terminal


def _camera_working_point_snapshot(point: CameraWorkingPoint) -> dict[str, object]:
    """Return the adapter readback as plain, archive-ready run metadata."""

    mode = getattr(point.acquisition_mode, "value", point.acquisition_mode)
    return {
        "acquisition_mode": str(mode),
        "frame_shape_yx": [int(value) for value in point.frame_shape_yx],
        "sensor_shape_yx": [int(value) for value in point.sensor_shape_yx],
        "roi_origin_yx": [int(value) for value in point.roi_origin_yx],
        "roi_shape_yx": [int(value) for value in point.roi_shape_yx],
        "binning_yx": [int(value) for value in point.binning_yx],
        "dtype": point.dtype.str,
        "count_unit": str(point.count_unit),
        "offset_counts": (
            None if point.offset_counts is None else float(point.offset_counts)
        ),
        "electrons_per_count": (
            None
            if point.electrons_per_count is None
            else float(point.electrons_per_count)
        ),
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
        "readout_mode": str(point.readout_mode),
    }


@dataclass(frozen=True)
class CameraMeasurementRequest:
    """One frozen camera selection and acquisition working point."""

    camera_key: str
    exposure_seconds: float
    roi_xywh: tuple[int, int, int, int] | None
    repeat: int
    frames_per_cycle: int
    #: Read the camera in photoelectrons instead of counts, through the
    #: conversion the CAMERA's configuration states.  A camera that states
    #: none falls back to raw counts rather than inventing a conversion; the
    #: effective choice rides in the run record.
    photoelectrons: bool = True

    def __post_init__(self) -> None:
        camera_key = str(self.camera_key).strip()
        if not camera_key:
            raise ValueError("camera_key must be non-empty")
        exposure = float(self.exposure_seconds)
        if not np.isfinite(exposure) or exposure <= 0:
            raise ValueError("exposure_seconds must be positive and finite")
        repeat = int(self.repeat)
        frames_per_cycle = int(self.frames_per_cycle)
        if repeat < 0:
            raise ValueError("repeat must be non-negative")
        if frames_per_cycle <= 0:
            raise ValueError("frames_per_cycle must be positive")
        roi = self.roi_xywh
        if roi is not None:
            try:
                roi = tuple(int(value) for value in roi)
            except (TypeError, ValueError) as exc:
                raise TypeError("roi_xywh must contain four integers or be None") from exc
            if len(roi) != 4:
                raise ValueError("roi_xywh must contain four integers or be None")
            x, y, width, height = roi
            if x < 0 or y < 0 or width <= 0 or height <= 0:
                raise ValueError("roi_xywh must have a non-negative origin and positive size")
        object.__setattr__(self, "camera_key", camera_key)
        object.__setattr__(self, "exposure_seconds", exposure)
        object.__setattr__(self, "roi_xywh", roi)
        object.__setattr__(self, "repeat", repeat)
        object.__setattr__(self, "frames_per_cycle", frames_per_cycle)
        object.__setattr__(self, "photoelectrons", bool(self.photoelectrons))




@dataclass(frozen=True)
class MeasurementResult:
    cycles: tuple[tuple[CameraFrameRecord, ...], ...]
    cycle_count: int
    snapshot: OwnedSnapshot
    publication: SignalPublication
    terminal: CameraCaptureTerminalRecord

    @property
    def frames(self) -> tuple[CameraFrameRecord, ...]:
        return tuple(frame for cycle in self.cycles for frame in cycle)


class FiniteCapture:
    """An armed finite capture whose triggers are supplied by another owner."""

    def __init__(
        self,
        node: "CameraMeasurementNode",
        *,
        repeat: int,
        frames_per_cycle: int,
        timeout: float,
        owns_generation: bool,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        self.node = node
        self.camera = node.camera
        self.repeat = int(repeat)
        self.frames_per_cycle = int(frames_per_cycle)
        self.timeout = float(timeout)
        self.owns_generation = bool(owns_generation)
        #: Asked between reads, and only by a caller that has someone to ask
        #: -- a hosted run has the host's cancel, a notebook has nobody.
        self.should_stop = should_stop
        self.closed = False
        self.collected: MeasurementResult | None = None
        self.completed_cycles = 0
        #: Whether the owner asked this capture to stop, which is the one
        #: reason a device may honestly report more frames than the completed
        #: cycles account for: the cycle it was walking away from.
        self.stopped = False
        self.terminal: CameraCaptureTerminalRecord | None = None
        self._pending_records: list[CameraFrameRecord] = []

    def collect(
        self,
        *,
        commit_cycle: Callable[[tuple[CameraFrameRecord, ...], int], None]
        | None = None,
        retain_cycles: bool | None = None,
    ) -> MeasurementResult | None:
        """Read the armed cycles and publish them.

        ``commit_cycle`` receives only the newly completed cycle and its run
        index.  Runtime owns every prior cycle and the fixed authored shape;
        handing the whole prefix back to a plugin is the O(N^2) path this
        method replaces. A normal Stop first stops intake, then commits every
        accepted complete cycle before sealing. A final partial cycle is
        counted but never published as complete. Failures do not publish a
        remaining queue.
        """

        if self.closed:
            raise RuntimeError("finite capture is closed")
        if self.collected is not None:
            return self.collected
        if commit_cycle is None:
            if not self.owns_generation:
                raise TypeError("hosted finite capture requires commit_cycle")
            commit_cycle = self.node._commit_direct_cycle
        if not callable(commit_cycle):
            raise TypeError("commit_cycle must be callable")
        keep = self.owns_generation if retain_cycles is None else bool(retain_cycles)
        retained: list[tuple[CameraFrameRecord, ...]] = []
        try:
            for index in range(self.repeat):
                cycle = self.next_cycle()
                if cycle is None:
                    break
                commit_cycle(cycle, index)
                if keep:
                    retained.append(cycle)
            terminal = self.close()
            # Stop first fixes the accepted prefix. Keep every complete cycle
            # from it, including a cycle partly read when Stop arrived.
            complete = min(self.repeat, terminal.produced_count // self.frames_per_cycle)
            while self.completed_cycles < complete:
                pending = self._pending_records
                pending.extend(self.node.read_records(
                    self.frames_per_cycle - len(pending), timeout=0.0, exact=True,
                ))
                cycle = _strict_cycle_ordinals(
                    pending, expected_start=self.completed_cycles * self.frames_per_cycle,
                    frames_per_cycle=self.frames_per_cycle,
                )
                pending.clear()
                index = self.completed_cycles
                self.completed_cycles += 1
                commit_cycle(cycle, index)
                if keep:
                    retained.append(cycle)
            remaining = terminal.produced_count - self.node._next_record_ordinal
            if remaining:
                self.node.read_records(remaining, timeout=0.0, exact=True)
            self._pending_records.clear()
            terminal = replace(terminal, no_more_frames=True)
            self.terminal = terminal
        except BaseException as error:
            if not self.closed:
                try:
                    self.camera.finish_record_capture()
                except BaseException as cleanup:
                    error.add_note(f"camera cleanup also failed: {cleanup}")
                finally:
                    self.closed = True
            if self.owns_generation:
                self.node.signal_plane.retire(self.node)
            raise
        if not self.completed_cycles:
            if self.owns_generation:
                self.node.signal_plane.retire(self.node)
            return None
        if self.owns_generation:
            self.node.signal_plane.seal_committed(
                self.node,
                cut_short=self.completed_cycles < self.repeat,
            )
        publication = self.node.signal_plane.latest_publication(
            self.node.signal_key(CAMERA_FRAMES_OUTPUT.name)
        )
        if not isinstance(publication, SignalPublication):
            raise RuntimeError("signal plane did not retain the camera commit")
        snapshot = self.node.signal_plane.current_dataset(
            self.node.signal_key(CAMERA_FRAMES_OUTPUT.name),
            publication,
        )
        self.collected = MeasurementResult(
            tuple(retained),
            self.completed_cycles,
            snapshot,
            publication,
            terminal,
        )
        return self.collected

    def next_cycle(self) -> tuple[CameraFrameRecord, ...] | None:
        """The next complete cycle of this capture, or None if asked to stop.

        A scan takes its cycles one at a time -- that is what lets the dataset
        grow on screen while the board is still playing -- and so does a
        finite measurement.  Same read.

        The wait is sliced.  A cycle is complete only when all of its frames
        have arrived, and how long that may take is the CAMERA's timeout (2 s
        virtual, 10 s on the qCMOS) -- but a Stop is only ever seen between
        reads, so waiting that out in one call is a Stop the operator watches
        the console refuse for ten seconds.  The deadline is unchanged; only
        the granularity is.
        """

        if self.closed:
            raise RuntimeError("finite capture is closed")
        records = self._pending_records
        deadline = monotonic() + self.timeout
        while len(records) < self.frames_per_cycle:
            if self.should_stop is not None and self.should_stop():
                self.stopped = True
                return None
            arrived = self.node.read_records(
                self.frames_per_cycle - len(records),
                timeout=min(_CANCEL_RESPONSE_SECONDS, max(0.0, deadline - monotonic())),
                exact=False,
            )
            if arrived:
                records.extend(arrived)
                # A frame arrived, so the camera is delivering: the deadline
                # is how long a FRAME may take, not how long a cycle may.
                deadline = monotonic() + self.timeout
                continue
            if monotonic() >= deadline:
                break
        if len(records) != self.frames_per_cycle:
            point = self.node.actual_working_point
            exposure = "unknown" if point is None else format(point.exposure_seconds, "g")
            interval = None if point is None else point.required_external_trigger_interval_seconds
            raise RuntimeError(
                f"the camera returned {len(records)} frame(s) of a "
                f"{self.frames_per_cycle}-frame cycle before timeout. Camera "
                f"readback: exposure {exposure}s, minimum external-trigger "
                "interval "
                f"{'unknown' if interval is None else format(interval, 'g') + 's'}"
                ". Camera Measurement has no Pulse schedule; inspect the "
                "external rising-edge count, spacing, and source state"
            )
        expected = self.completed_cycles * self.frames_per_cycle
        cycle = _strict_cycle_ordinals(
            records,
            expected_start=expected,
            frames_per_cycle=self.frames_per_cycle,
        )
        self.completed_cycles += 1
        records.clear()
        return cycle

    def close(self) -> CameraCaptureTerminalRecord:
        """Finish the capture with the device's evidence, or not at all.

        Only a finish the device completed and the count check accepted is
        cached; a finish the device refused leaves the capture open so the
        next close retries the same finish.  A capture that a failed collect
        already closed has no terminal to hand out and says so.
        """

        if self.terminal is not None:
            return self.terminal
        if self.closed:
            raise RuntimeError("finite capture closed without terminal evidence")
        terminal = _strict_terminal(
            self.camera.finish_record_capture(),
            expected_frames=self.completed_cycles * self.frames_per_cycle,
            stopped=self.stopped,
        )
        self.closed = True
        self.terminal = terminal
        return terminal


class MonitorCapture:
    """A repeat-zero capture publishing every accepted complete cycle."""

    def __init__(
        self,
        camera: CameraAdapter,
        *,
        node: "CameraMeasurementNode",
        owns_generation: bool,
        commit_live: Callable[..., Mapping[str, SignalValue]] | None,
    ) -> None:
        self.camera = camera
        self.node = node
        self.owns_generation = bool(owns_generation)
        self.closed = False
        self.terminal: CameraCaptureTerminalRecord | None = None
        self.latest_record: CameraFrameRecord | None = None
        self._pending_records: list[CameraFrameRecord] = []
        self._revision = 0
        if self.owns_generation:
            if commit_live is not None:
                raise ValueError("a direct monitor cannot use a host commit function")
            self._commit_live = self.node._commit_direct_outputs
        else:
            if not callable(commit_live):
                raise TypeError("a hosted monitor requires commit_live")
            self._commit_live = commit_live

    def poll(self) -> CameraFrameRecord | None:
        if self.closed:
            raise RuntimeError("monitor capture is closed")
        records = self.node.read_records(
            1, timeout=_CANCEL_RESPONSE_SECONDS, exact=False
        )
        if not records:
            return None
        for record in records:
            self._accept_record(record)
        self.latest_record = records[-1]
        return self.latest_record

    def _accept_record(self, record: CameraFrameRecord) -> None:
        """Publish only a physically aligned, contiguous camera cycle."""

        cycle_size = self.node.frames_per_cycle
        ordinal = int(record.source_ordinal)
        pending = self._pending_records
        expected = self._revision * cycle_size + len(pending)
        if ordinal != expected:
            raise RuntimeError(f"camera frame sequence gap: expected {expected}, received {ordinal}")
        pending.append(record)
        if len(pending) == cycle_size:
            cycle = _strict_cycle_ordinals(
                pending,
                expected_start=int(pending[0].source_ordinal),
                frames_per_cycle=cycle_size,
            )
            pending.clear()
            self._revision += 1
            self._commit_live(
                {
                    CAMERA_FRAMES_OUTPUT.name: _monitor_cycle_output(
                        self.node,
                        cycle,
                        self._revision,
                    )
                }
            )

    def close(self, *, drain: bool = True) -> CameraCaptureTerminalRecord:
        """Disarm the camera, then detach the generation.

        Only the terminal the device produced is cached and handed back: a
        disarm the device refused is raised and left for the next close to
        retry, never answered with a made-up all-clear over a camera that is
        still armed.  A direct monitor detaches its own generation once the
        device has stopped; a hosted monitor leaves detachment and slot
        closing to its host, which keeps owning the plane generation through
        worker termination.  The device disarm comes first, so a detach
        failure cannot skip it.
        """

        if self.terminal is not None:
            return self.terminal
        terminal = self.camera.finish_record_capture()
        self.closed = True
        if drain:
            remaining = terminal.produced_count - self.node._next_record_ordinal
            for _ in range(remaining):
                record, = self.node.read_records(1, timeout=0.0, exact=True)
                self._accept_record(record)
            # A partial cycle is counted but is not a scientific publication.
            self._pending_records.clear()
            terminal = replace(terminal, no_more_frames=True)
        self.terminal = terminal
        if self.owns_generation:
            if self._revision:
                self.node.signal_plane.seal_committed(self.node)
            else:
                self.node.signal_plane.retire(self.node)
        return terminal


class CameraMeasurementNode:
    """Commit each atomic camera cycle to one stable ``frames`` signal."""

    def __init__(
        self,
        *,
        camera: CameraAdapter,
        request: CameraMeasurementRequest,
        signal_plane: object,
        producer: str = "camera_measurement",
    ) -> None:
        if not isinstance(camera, CameraAdapter):
            raise TypeError("camera must implement CameraAdapter")
        if not isinstance(request, CameraMeasurementRequest):
            raise TypeError("request must be CameraMeasurementRequest")
        self.camera = camera
        self._request = request
        self._actual_working_point: CameraWorkingPoint | None = None
        self._run_record: dict[str, object] | None = None
        self._settings_session_id: str | None = None
        if signal_plane is None:
            raise TypeError("signal_plane must be supplied by the runtime owner")
        self.signal_plane = signal_plane
        self.instance_id = str(producer).strip()
        if not self.instance_id:
            raise ValueError("producer must be non-empty")
        self._generation: object | None = None

    @property
    def request(self) -> CameraMeasurementRequest:
        return self._request

    @property
    def actual_working_point(self) -> CameraWorkingPoint | None:
        return self._actual_working_point

    @property
    def reads_photoelectrons(self) -> bool:
        point = self._actual_working_point
        if point is None:
            raise RuntimeError("camera working point is not frozen")
        return bool(
            self.request.photoelectrons
            and point.electrons_per_count is not None
        )

    @property
    def frame_value_unit(self) -> str | None:
        point = self._actual_working_point
        if point is None:
            raise RuntimeError("camera working point is not frozen")
        return None if self.reads_photoelectrons else point.count_unit

    @property
    def generation(self) -> object:
        if self._generation is None:
            raise RuntimeError("camera acquisition has no active generation")
        return self._generation

    @property
    def camera_key(self) -> str:
        return self.request.camera_key

    @property
    def exposure_seconds(self) -> float:
        return self.request.exposure_seconds

    @property
    def roi_xywh(self) -> tuple[int, int, int, int] | None:
        return self.request.roi_xywh

    @property
    def repeat(self) -> int:
        return self.request.repeat

    @property
    def frames_per_cycle(self) -> int:
        return self.request.frames_per_cycle

    @property
    def dataset_output_declarations(self) -> tuple[DatasetOutputDeclaration, ...]:
        return (CAMERA_FRAMES_OUTPUT,)

    def signal_key(self, output_name: str) -> str:
        name = str(output_name)
        if name not in {
            declaration.name for declaration in self.dataset_output_declarations
        }:
            raise KeyError(f"unknown camera output {output_name!r}")
        return f"@logic/{self.instance_id}/{name}"

    def _configure_for_run(self) -> CameraWorkingPoint:
        self._actual_working_point = None
        self._run_record = None
        self._next_record_ordinal = 0
        # This measurement owns both: it exists to point the camera.  The
        # geometry first, because it is the expensive one to get wrong.
        self.camera.set_roi(self.request.roi_xywh)
        point = self.camera.set_exposure_seconds(self.request.exposure_seconds)
        if not isinstance(point, CameraWorkingPoint):
            raise TypeError("camera set_exposure_seconds must return CameraWorkingPoint")
        return point

    def _buffer_frame_count(self, point: CameraWorkingPoint) -> int:
        # SDK ring and accepted FIFO each hold at most N raw frames. Scientific
        # history and one-frame copy scratch are not receive-buffer storage.
        frame_bytes = int(np.prod(point.frame_shape_yx)) * point.dtype.itemsize
        count = _RECEIVE_BUFFER_MIB * 1024 * 1024 // (2 * frame_bytes)
        if count < self.frames_per_cycle:
            raise ValueError(
                f"Internal receive buffer {_RECEIVE_BUFFER_MIB} MiB cannot hold "
                f"one {self.frames_per_cycle}-frame cycle at {frame_bytes} bytes/frame"
            )
        return count

    def _configure_capture(self) -> CameraWorkingPoint:
        """Apply and freeze the requested working point without arming."""

        # ``set_exposure_seconds`` already returns the authoritative readback
        # after ROI and exposure have been applied.  Reading the complete
        # qCMOS property surface again made every finite Start pay a third
        # full working-point query before ARM.
        self._freeze_working_point(self._configure_for_run())
        assert self._actual_working_point is not None
        return self._actual_working_point

    def _arm_configured(
        self,
        *,
        owns_generation: bool,
        should_stop: Callable[[], bool] | None,
    ) -> FiniteCapture:
        """Arm a capture whose hardware working point is already frozen."""

        if self._actual_working_point is None or self._run_record is None:
            raise RuntimeError("camera capture must be configured before arm")
        timeout = float(self.camera.timeout)
        total = self.request.repeat * self.request.frames_per_cycle
        groups = (self.request.frames_per_cycle,) * self.request.repeat
        buffer_frames = self._buffer_frame_count(self._actual_working_point)
        self.camera.arm(
            total,
            source_group_sizes=groups,
            buffer_frame_count=buffer_frames,
            timeout=timeout,
        )
        self._freeze_working_point(self.camera.working_point())
        self._run_record["acquisition"] = {"buffer_frame_count": buffer_frames}
        return FiniteCapture(
            self,
            repeat=self.request.repeat,
            frames_per_cycle=self.request.frames_per_cycle,
            timeout=timeout,
            owns_generation=owns_generation,
            should_stop=should_stop,
        )

    def read_records(
        self,
        count: int,
        *,
        timeout: float,
        exact: bool,
    ) -> tuple[CameraFrameRecord, ...]:
        """Take frames from the camera, in the unit this run publishes.

        THE read.  A capture never touches the adapter itself, and that is
        the whole point of this method existing: when the finite read did the
        conversion and the monitor read did not, a live panel showed counts
        while the same run's saved samples were electrons, and nothing on
        screen said which was which.  One intake cannot disagree with itself.

        Counts stay the sensor's own integers -- the pipeline is built on
        that, and a 2048x2048 frame costs 9.6 ms and twice the memory to
        widen.  Photoelectrons are float32: the conversion is affine, so it
        moves no decision (thresholds move with it), and what it buys is
        numbers a physicist can read.
        """

        records = tuple(
            self.camera.read_frame_records(int(count), timeout=timeout, exact=exact)
        )
        for record in records:
            if record.source_ordinal != self._next_record_ordinal:
                raise RuntimeError(
                    f"camera frame sequence gap: expected {self._next_record_ordinal}, "
                    f"received {record.source_ordinal}"
                )
            self._next_record_ordinal += 1
        if not self.reads_photoelectrons:
            return records
        point = self._actual_working_point
        assert point is not None and point.electrons_per_count is not None
        offset = np.float32(point.offset_counts)
        scale = np.float32(point.electrons_per_count)
        return tuple(
            replace(
                record,
                image=(np.asarray(record.image, dtype=np.float32) - offset) * scale,
            )
            for record in records
        )

    def _freeze_working_point(self, point: CameraWorkingPoint) -> None:
        if not isinstance(point, CameraWorkingPoint):
            raise TypeError("camera working_point must return CameraWorkingPoint")
        self._actual_working_point = point
        photoelectrons = self.reads_photoelectrons
        record = {
            "node": self.instance_id,
            "parameters": {
                "exposure_seconds": self.request.exposure_seconds,
                "roi_xywh": (
                    None
                    if self.request.roi_xywh is None
                    else list(self.request.roi_xywh)
                ),
                "repeat": self.request.repeat,
                "frames_per_cycle": self.request.frames_per_cycle,
                PHOTOELECTRONS: photoelectrons,
            },
            "named_devices": {"camera": self.request.camera_key},
            "device_snapshots": {
                "camera": _camera_working_point_snapshot(point),
            },
        }
        self._run_record = record
        self._settings_session_id = None

    def _camera_event_record(
        self,
        records: Sequence[CameraFrameRecord],
    ) -> dict[str, object]:
        """The settings frozen on THESE frames, never current state.

        An event record describes the chunk it travels with -- this cycle,
        these frames -- and nothing before it: the Runtime merges the epoch
        ranges of every retained chunk into the canonical prefix, so a cycle
        that also carried the epochs of earlier cycles was the same fact stated
        twice, once wrongly.  What does span the acquisition is the device
        session: a camera that changed identity between cycles is refused.
        """

        frames = tuple(records)
        references = tuple(
            (record.settings_session_id, record.settings_epochs)
            for record in frames
            if record.settings_session_id is not None
        )
        if not references:
            return {}
        if len(references) != len(frames):
            raise RuntimeError("one camera cycle mixed settings-aware and unaware frames")
        session_ids = {str(session_id) for session_id, _epochs in references}
        if len(session_ids) != 1:
            raise RuntimeError("camera device session changed inside one cycle")
        session_id = next(iter(session_ids))
        if self._settings_session_id not in (None, session_id):
            raise RuntimeError("camera device session changed during one acquisition")
        self._settings_session_id = session_id
        ordered = sorted({
            epoch
            for _session_id, epochs in references
            for epoch in epochs
        })
        ranges: list[list[int]] = []
        for epoch in ordered:
            if ranges and epoch == ranges[-1][1] + 1:
                ranges[-1][1] = epoch
            else:
                ranges.append([epoch, epoch])
        return {"device_settings": {
            "camera": {
                "device_session_id": session_id,
                "epoch_ranges": ranges,
                "mixed": len(ordered) > 1,
            }
        }}

    @property
    def run_record(self) -> dict[str, object]:
        """What this acquisition IS, frozen when the camera was armed."""

        record = self._run_record
        if record is None:
            raise RuntimeError("camera run record was not frozen after arm")
        return record

    def _commit_direct_outputs(
        self,
        outputs: Mapping[str, LiveDatasetOutput],
    ) -> Mapping[str, SignalValue]:
        return self.signal_plane.commit_live(self, outputs)

    def _commit_direct_cycle(
        self,
        cycle: tuple[CameraFrameRecord, ...],
        index: int,
    ) -> None:
        if index == 0:
            self._run_record = self.signal_plane.set_run_record(self, self.run_record)
        self._commit_direct_outputs(
            {
                CAMERA_FRAMES_OUTPUT.name: _finite_cycle_output(
                    self,
                    cycle,
                    index,
                )
            }
        )

    def prepare(
        self,
        *,
        owns_generation: bool = True,
        should_stop: Callable[[], bool] | None = None,
    ) -> FiniteCapture:
        """Arm the camera for one acquisition.

        The frozen request is the acquisition this node performs.
        ``owns_generation`` is False when a NodeHost has already begun the
        generation for us, and ``should_stop`` is that host's cancel: a
        capture asks it between reads, which is the only place a blocking
        read can be interrupted.

        The publisher declares run metadata before its first commit; a task
        may finish static geometry from that first frame before declaring it.
        """

        if self.request.repeat <= 0:
            raise ValueError("finite prepare requires request.repeat greater than zero")
        owns_generation = bool(owns_generation)
        if owns_generation:
            self._generation = self.signal_plane.begin_generation(self)
        try:
            self._configure_capture()
            capture = self._arm_configured(
                owns_generation=owns_generation,
                should_stop=should_stop,
            )
            return capture
        except BaseException as error:
            try:
                self.camera.finish_record_capture()
            except BaseException as cleanup:
                error.add_note(f"camera cleanup also failed: {cleanup}")
            if owns_generation:
                self.signal_plane.retire(self)
            raise

    def execute(self, context: object) -> dict[str, object]:
        """Hosted entry point: the same acquisition, published through the host.

        A NodeHost has already begun the generation, and publications must go
        through its context so the host can observe them.  Everything else --
        arming, reading cycles, building the snapshot -- is the identical code
        the notebook path runs, because a second implementation is how a virtual
        bench and a real one start to disagree.
        """

        self._generation = context.generation
        if self.repeat == 0:
            capture = self.monitor(
                owns_generation=False,
                commit_live=context.commit_live,
            )
            try:
                self._run_record = context.set_run_record(self.run_record)
                context.report_ready()
                while not context.cancel_requested():
                    capture.poll()
            except BaseException as error:
                try:
                    capture.close(drain=False)
                except BaseException as cleanup:
                    error.add_note(f"camera cleanup also failed: {cleanup}")
                raise
            capture.close()
            return {
                "signals": tuple(
                    self.signal_key(value.name)
                    for value in self.dataset_output_declarations
                )
            }
        capture = self.prepare(
            owns_generation=False,
            should_stop=context.cancel_requested,
        )
        try:
            self._run_record = context.set_run_record(self.run_record)
            context.report_ready()
        except BaseException:
            # Stop can arrive between the completed arm and its acknowledgement.
            # collect() has not taken ownership of cleanup yet.
            capture.stopped = True
            capture.close()
            raise

        def commit_cycle(cycle: object, index: int) -> None:
            context.commit_live(
                {CAMERA_FRAMES_OUTPUT.name: _finite_cycle_output(self, cycle, index)}
            )
            # A cycle landed is a repeat played; the bar says so as it happens.
            context.report_progress(
                "Capturing", current=index + 1, total=int(self.repeat)
            )

        result = capture.collect(commit_cycle=commit_cycle, retain_cycles=False)
        return {
            "cycles": 0 if result is None else result.cycle_count,
            "signals": tuple(
                self.signal_key(value.name)
                for value in self.dataset_output_declarations
            ),
        }

    def monitor(
        self,
        *,
        owns_generation: bool = True,
        commit_live: Callable[..., Mapping[str, SignalValue]] | None = None,
    ) -> MonitorCapture:
        cycle_size = self.frames_per_cycle
        if self.request.repeat != 0:
            raise ValueError("monitor requires request.repeat equal to zero")
        owns_generation = bool(owns_generation)
        if owns_generation:
            if commit_live is not None:
                raise ValueError("a direct monitor cannot use a host commit function")
            self._generation = self.signal_plane.begin_generation(self)
        elif not callable(commit_live):
            raise TypeError("a hosted monitor requires commit_live")
        try:
            point = self._configure_for_run()
            buffer_frames = self._buffer_frame_count(point)
            # The camera's own timeout still governs ARMING -- how long the
            # device may take to become ready is the device's fact.
            timeout = float(self.camera.timeout)
            self.camera.arm(
                None,
                source_group_sizes=None if cycle_size == 1 else (cycle_size,),
                buffer_frame_count=buffer_frames,
                timeout=timeout,
            )
            self._freeze_working_point(self.camera.working_point())
            self._run_record["acquisition"] = {"buffer_frame_count": buffer_frames}
            if owns_generation:
                self._run_record = self.signal_plane.set_run_record(self, self.run_record)
            return MonitorCapture(
                self.camera,
                node=self,
                owns_generation=owns_generation,
                commit_live=commit_live,
            )
        except BaseException as error:
            try:
                self.camera.finish_record_capture()
            except BaseException as cleanup:
                error.add_note(f"camera cleanup also failed: {cleanup}")
            if owns_generation:
                self.signal_plane.retire(self)
            raise


__all__ = [
    "CAMERA_FRAMES_OUTPUT",
    "frames_snapshot",
    "CameraMeasurementNode",
    "CameraMeasurementRequest",
    "FiniteCapture",
    "MeasurementResult",
    "MonitorCapture",
]
