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
    OwnedSnapshot,
    PointColumn,
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


_CAMERA_FRAME_CONTRACT = "camera.frames.v1"

#: The camera's ONE output: a cycle of frames on the dataset's POINT axis.
#: A cycle is one acquisition event; publishing its frames as N sibling
#: signals leaked the acquisition configuration into the signal vocabulary
#: (a panel bound to "frame_1" broke the moment frames_per_cycle changed),
#: and no consumer could see the whole cycle at once.  The frames live on
#: the POINT axis -- (repeat cycles) x (frame points) x (y, x) -- because a
#: frame is a point of the acquisition, not structure inside a pixel plane:
#: that is what lets a grid facet the frames side by side and lets a scan
#: nest them into its own point table.  The column's role is READOUT_EVENT,
#: the point-domain role the data model reserves for exactly this.
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


def _frame_point_column(producer: str, frames: int) -> PointColumn:
    """The frame-index point column one cycle publishes."""

    return PointColumn(
        AxisId(f"{producer}.frames.frame"),
        "frame",
        READOUT_EVENT,
        PointColumn.NUMERIC,
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
):
    """The sensor pixels a frame covers, as its two spatial axes.

    A frame is a crop of a sensor, and the numbers that mean anything about it
    are the sensor's own pixel coordinates: where the traps are, where an ROI
    starts, what a region drawn on the picture refers to.  Published without
    axes the picture was indexed from zero instead, so every coordinate on it
    was an offset into that particular crop -- the same region meant somewhere
    else the moment the ROI moved, which is exactly what a region does when it
    is used to set the ROI.

    The frame axis is a point column, so the cell is (y, x): those are the two
    cell axes the dataset will generate, in that order.
    """

    if point is None:
        return None
    origin_y, origin_x = (int(value) for value in point.roi_origin_yx)
    step_y, step_x = (int(value) for value in getattr(point, "binning_yx", (1, 1)))
    height, width = (int(value) for value in shape_yx)
    return {
        SPATIAL_Y: _sensor_pixel_axis(
            producer, 0, SPATIAL_Y, origin_y, max(1, step_y), height
        ),
        SPATIAL_X: _sensor_pixel_axis(
            producer, 1, SPATIAL_X, origin_x, max(1, step_x), width
        ),
    }


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
        np.stack(
            [
                np.stack([np.asarray(record.image) for record in cycle], axis=0)
                for cycle in frames
            ],
            axis=0,
        ),
        producer=producer,
        signal=CAMERA_FRAMES_OUTPUT.name,
        roles=(READOUT_EVENT, SPATIAL_Y, SPATIAL_X),
        axis_specs=_pixel_axes(
            working_point,
            np.asarray(frames[0][0].image).shape,
            producer=producer,
        ),
        point_columns={READOUT_EVENT: _frame_point_column(producer, len(frames[0]))},
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
    canonical = replace(
        event.block.schema,
        repeat_axis=replace(
            event.block.schema.repeat_axis,
            size=node.repeat,
        ),
    )
    frames = node.frames_per_cycle
    return LiveDatasetOutput(
        CAMERA_FRAMES_OUTPUT,
        event,
        DatasetCoverage((index + 1) * frames, node.repeat * frames),
        node.run_record,
        canonical,
        (index, 0),
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
        node.run_record,
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
) -> CameraCaptureTerminalRecord:
    if (
        terminal.produced_count != expected_frames
        or not terminal.source_stopped
        or not terminal.no_more_frames
        or not terminal.joined
    ):
        raise RuntimeError(
            "camera did not prove exact capture completion: "
            f"expected {expected_frames} frame(s), produced "
            f"{terminal.produced_count}"
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


class CameraCycleSource:
    """The point's value is the cycle of frames the fired program triggered.

    The camera is armed once for the whole table -- one buffer, one exact
    count -- because the board plays the table from one fire and every cycle's
    triggers are already scheduled.  Reading a cycle at a time is what lets
    the scan grow on screen while it runs.

    There is no backlog to discard: a triggered camera holds exactly the
    frames the program asked for, so arming IS the moment the old world ends.
    """

    def __init__(self, camera_node: object) -> None:
        self.camera_node = camera_node
        self._generation: object | None = None
        self._capture = None
        self._taken = 0
        self._should_stop: Callable[[], bool] | None = None
        self._cycles: int | None = None

    def open(self, context: object, *, cycles: int) -> None:
        # The frames belong to the run that opened this source, so they are
        # stamped with its generation -- not with one this source invented.
        self._generation = context.generation
        # ... and so does its cancel: the capture asks it between reads,
        # which is where a scan waiting on a trigger that never came can
        # actually be stopped.
        self._should_stop = context.cancel_requested
        self._taken = 0
        # The camera was built for a definite acquisition; if the plan plays a
        # different number of cycles, one of the two is wrong about what this
        # run is, and an exact capture would deadlock waiting for frames that
        # were never triggered.
        request = self.camera_node.request
        if int(cycles) != request.repeat:
            raise ValueError(
                f"the plan plays {int(cycles)} cycles and the camera is armed "
                f"for {request.repeat}"
            )
        self._cycles = int(cycles)

    def validate(
        self,
        program: object,
        table: object = None,
        *,
        cycles: int | None = None,
    ) -> None:
        """Freeze the camera working point before the plan arms devices."""

        if self._cycles is None:
            raise RuntimeError("the camera cycle source was not opened")
        del program, table, cycles
        if self._capture is None:
            self.camera_node._configure_capture()

    def arm(self) -> None:
        """Arm the already-configured camera once for the whole run."""

        if self._cycles is None:
            raise RuntimeError("the camera cycle source was not opened")
        if self._capture is None:
            try:
                self._capture = self.camera_node._arm_configured(
                    owns_generation=False,
                    should_stop=self._should_stop,
                )
            except BaseException:
                self.camera_node.camera.finish_record_capture()
                raise

    def next_value(self, context: object) -> SignalValue:
        if self._capture is None:
            raise RuntimeError("the camera source was not armed")
        if context.cancel_requested():
            raise RuntimeError("the scan was cancelled")
        records = self._capture.next_cycle()
        if records is None:
            raise RuntimeError("the scan was cancelled")
        self._taken += 1
        return SignalValue(
            CAMERA_FRAMES_OUTPUT.name,
            frames_snapshot(
                (records,),
                producer=self.camera_node.instance_id,
                generation=self._generation,
                revision=self._taken,
                working_point=self.camera_node.actual_working_point,
                value_unit=self.camera_node.frame_value_unit,
            ),
            MonitorCoverage(written_cells=len(records), total_cells=len(records)),
            run_record=self.camera_node.run_record,
        )

    def close(self) -> CameraCaptureTerminalRecord | None:
        capture, self._capture = self._capture, None
        if capture is None:
            return None
        return capture.close()

    def describe(self) -> dict[str, object]:
        request = self.camera_node.request
        return {
            "source_camera": request.camera_key,
            "frames_per_cycle": request.frames_per_cycle,
        }


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
        self.terminal: CameraCaptureTerminalRecord | None = None

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
        method replaces.  A run
        that is asked to stop keeps the cycles it took -- they were measured
        -- while one stopped before its first cycle has nothing to publish.
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
            terminal = self.camera.finish_record_capture()
            _strict_terminal(
                terminal,
                expected_frames=self.completed_cycles * self.frames_per_cycle,
            )
            self.terminal = terminal
        except BaseException:
            self.camera.finish_record_capture()
            self.closed = True
            if self.owns_generation:
                self.node.signal_plane.retire(self.node)
            raise
        self.closed = True
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
        records: tuple[CameraFrameRecord, ...] = ()
        deadline = monotonic() + self.timeout
        while len(records) < self.frames_per_cycle:
            if self.should_stop is not None and self.should_stop():
                return None
            arrived = self.node.read_records(
                self.frames_per_cycle - len(records),
                timeout=min(_CANCEL_RESPONSE_SECONDS, max(0.0, deadline - monotonic())),
                exact=False,
            )
            if arrived:
                records += tuple(arrived)
                # A frame arrived, so the camera is delivering: the deadline
                # is how long a FRAME may take, not how long a cycle may.
                deadline = monotonic() + self.timeout
                continue
            if monotonic() >= deadline:
                break
        if len(records) != self.frames_per_cycle:
            # The two numbers that decide this, from the sensor rather than
            # from a guess: how long it integrates per trigger, and how often
            # it will accept one.  A trigger that arrives while it is still
            # busy is ignored, which is one frame short of a cycle, every
            # cycle -- and "an incomplete atomic cycle" named none of that.
            point = self.node.actual_working_point
            exposure = "unknown" if point is None else format(point.exposure_seconds, "g")
            interval = None if point is None else point.required_external_trigger_interval_seconds
            raise RuntimeError(
                f"the camera returned {len(records)} frame(s) of a "
                f"{self.frames_per_cycle}-frame cycle: it integrates {exposure}s "
                "per trigger and accepts one only every "
                f"{'unknown' if interval is None else format(interval, 'g') + 's'}"
                ", and a trigger arriving before that is ignored -- the pulse "
                "must space its camera windows by more than that, or the "
                "exposure must come down"
            )
        expected = self.completed_cycles * self.frames_per_cycle
        cycle = _strict_cycle_ordinals(
            records,
            expected_start=expected,
            frames_per_cycle=self.frames_per_cycle,
        )
        self.completed_cycles += 1
        return cycle

    def close(self) -> CameraCaptureTerminalRecord:
        if self.closed:
            if self.terminal is None:
                raise RuntimeError("finite capture closed without terminal evidence")
            return self.terminal
        self.closed = True
        self.terminal = _strict_terminal(
            self.camera.finish_record_capture(),
            expected_frames=self.completed_cycles * self.frames_per_cycle,
        )
        return self.terminal


class MonitorCapture:
    """A repeat-zero monitor with latest-frame semantics."""

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
        if not pending:
            if ordinal % cycle_size:
                return
            pending.append(record)
        else:
            expected = pending[0].source_ordinal + len(pending)
            if ordinal != expected:
                pending.clear()
                if ordinal % cycle_size:
                    return
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

    def close(self) -> CameraCaptureTerminalRecord:
        """Release this capture and always disarm the camera.

        A direct monitor detaches its own generation.  A hosted monitor leaves
        detachment and slot closing to its host, which must keep owning the
        plane generation through worker termination.  Either way, a detach
        failure cannot skip the device disarm.
        """

        if self.closed:
            return CameraCaptureTerminalRecord(0, True, True, True)
        self.closed = True
        sealed: BaseException | None = None
        if self.owns_generation:
            try:
                if self._revision:
                    self.node.signal_plane.seal_committed(self.node)
                else:
                    self.node.signal_plane.retire(self.node)
            except BaseException as error:  # noqa: BLE001 - the camera still goes
                sealed = error
        terminal = self.camera.finish_record_capture()
        if sealed is not None:
            raise sealed
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
        # This measurement owns both: it exists to point the camera.  The
        # geometry first, because it is the expensive one to get wrong.
        self.camera.set_roi(self.request.roi_xywh)
        point = self.camera.set_exposure_seconds(self.request.exposure_seconds)
        if not isinstance(point, CameraWorkingPoint):
            raise TypeError("camera set_exposure_seconds must return CameraWorkingPoint")
        return point

    def _configure_capture(self) -> CameraWorkingPoint:
        """Apply and freeze the requested working point without arming."""

        self._configure_for_run()
        self._freeze_working_point(self.camera.working_point())
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
        self.camera.arm(
            total,
            source_group_sizes=groups,
            buffer_frame_count=total,
            timeout=timeout,
        )
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
        """

        if self.request.repeat <= 0:
            raise ValueError("finite prepare requires request.repeat greater than zero")
        owns_generation = bool(owns_generation)
        if owns_generation:
            self._generation = self.signal_plane.begin_generation(self)
        try:
            self._configure_capture()
            return self._arm_configured(
                owns_generation=owns_generation,
                should_stop=should_stop,
            )
        except BaseException:
            self.camera.finish_record_capture()
            if owns_generation:
                self.signal_plane.retire(self)
            raise

    def measure(self) -> MeasurementResult:
        """Collect externally triggered frames into one final publication."""

        result = self.prepare().collect()
        if result is None:
            raise RuntimeError("finite camera measurement ended without a cycle")
        return result

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
                while not context.cancel_requested():
                    capture.poll()
            finally:
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
        result = capture.collect(
            commit_cycle=lambda cycle, index: context.commit_live(
                {
                    CAMERA_FRAMES_OUTPUT.name: _finite_cycle_output(
                        self,
                        cycle,
                        index,
                    )
                }
            ),
            retain_cycles=False,
        )
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
        buffer_frames = 4 * cycle_size
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
            self._configure_for_run()
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
            return MonitorCapture(
                self.camera,
                node=self,
                owns_generation=owns_generation,
                commit_live=commit_live,
            )
        except BaseException:
            self.camera.finish_record_capture()
            if owns_generation:
                self.signal_plane.retire(self)
            raise


__all__ = [
    "CAMERA_FRAMES_OUTPUT",
    "CameraCycleSource",
    "frames_snapshot",
    "CameraMeasurementNode",
    "CameraMeasurementRequest",
    "FiniteCapture",
    "MeasurementResult",
    "MonitorCapture",
]
