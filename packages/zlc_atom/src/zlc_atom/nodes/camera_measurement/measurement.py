"""Finite exact and live-monitor camera capture shared by the node."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import numpy as np
from zlc_data import AxisId, PointColumn, READOUT_EVENT, SPATIAL_X, SPATIAL_Y
from zlc_runtime import MonitorCoverage
from zlc_runtime import (
    DatasetOutputDeclaration,
    FinalDatasetOutput,
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
from zlc_atom.data import snapshot_from_array


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


def _frame_point_column(producer: str, frames: int) -> PointColumn:
    """The frame-index point column one cycle publishes."""

    return PointColumn(
        AxisId(f"{producer}.frames.frame"),
        "frame",
        READOUT_EVENT,
        PointColumn.NUMERIC,
        tuple(range(int(frames))),
    )


def frames_snapshot(
    cycles: "Sequence[Sequence[CameraFrameRecord]]",
    *,
    producer: str,
    generation: object,
    revision: int,
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
        point_columns={READOUT_EVENT: _frame_point_column(producer, len(frames[0]))},
        generation=str(getattr(generation, "value", generation)),
        revision=int(revision),
    )


def _camera_working_point_snapshot(point: CameraWorkingPoint) -> dict[str, object]:
    """Return the adapter readback as plain, archive-ready run metadata."""

    mode = getattr(point.acquisition_mode, "value", point.acquisition_mode)
    return {
        "acquisition_mode": str(mode),
        "frame_shape_yx": tuple(int(value) for value in point.frame_shape_yx),
        "sensor_shape_yx": tuple(int(value) for value in point.sensor_shape_yx),
        "roi_origin_yx": tuple(int(value) for value in point.roi_origin_yx),
        "roi_shape_yx": tuple(int(value) for value in point.roi_shape_yx),
        "binning_yx": tuple(int(value) for value in point.binning_yx),
        "dtype": point.dtype.str,
        "count_unit": str(point.count_unit),
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


class _CameraMonitorSlot:
    """Application-owned live slot consumed by the runtime plane."""

    def __init__(self, node: "CameraMeasurementNode") -> None:
        self.node = node
        self.latest: tuple[CameraFrameRecord, ...] | None = None
        self.revision = 0
        self.closed = False
        self._change_listener: Callable[[], None] | None = None

    def set_change_listener(self, listener: Callable[[], None]) -> None:
        if self.closed:
            raise RuntimeError("camera monitor slot is closed")
        if not callable(listener):
            raise TypeError("camera monitor change listener must be callable")
        if self._change_listener is not None:
            raise RuntimeError("camera monitor slot already has a change listener")
        self._change_listener = listener

    def update(self, records: tuple[CameraFrameRecord, ...]) -> None:
        if self.closed:
            raise RuntimeError("camera monitor slot is closed")
        listener = self._change_listener
        if listener is None:
            raise RuntimeError("camera monitor slot is not attached")
        if len(records) != self.node.frames_per_cycle:
            raise ValueError("camera monitor update must contain one complete cycle")
        self.latest = records
        self.revision += 1
        listener()

    def freeze_live_outputs(self) -> dict[str, LiveDatasetOutput]:
        if self.closed:
            raise RuntimeError("camera monitor slot is closed")
        records = self.latest
        if records is None:
            raise RuntimeError("camera monitor slot has no accepted cycle")
        generation, revision = self.node._next_publication_stamp()
        # Cells are repeat x point rows; a cycle's frames ARE its point rows,
        # and a monitor publication is always one COMPLETE cycle.
        coverage = MonitorCoverage(
            written_cells=len(records),
            total_cells=len(records),
        )
        return {
            CAMERA_FRAMES_OUTPUT.name: LiveDatasetOutput(
                CAMERA_FRAMES_OUTPUT,
                frames_snapshot(
                    (records,),
                    producer=self.node.instance_id,
                    generation=generation,
                    revision=revision,
                ),
                coverage,
                self.node.run_record,
            )
        }

    def close(self) -> None:
        self.closed = True
        self._change_listener = None


class CameraCycleSource:
    """The point's value is the cycle of frames the fired program triggered.

    The camera is armed once for the whole table -- one buffer, one exact
    count -- because the board plays the table from one fire and every cycle's
    triggers are already scheduled.  Reading a cycle at a time is what lets
    the scan grow on screen while it runs.

    There is no backlog to discard: a triggered camera holds exactly the
    frames the program asked for, so arming IS the moment the old world ends.
    """

    def __init__(self, camera_node: object, *, trigger_channel: str) -> None:
        self.camera_node = camera_node
        self.trigger_channel = str(trigger_channel)
        self._generation: object | None = None
        self._capture = None
        self._taken = 0

    def open(self, context: object, *, cycles: int) -> None:
        # The frames belong to the run that opened this source, so they are
        # stamped with its generation -- not with one this source invented.
        self._generation = context.generation
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

    def arm(self, program: object, table: object = None) -> None:
        """Check the cycle, then arm for the whole run.

        The cycle the board is about to play must open exactly the windows
        this capture reads, or an exact read waits forever for a frame no
        trigger will ever produce.  ``table`` is the first row the board will
        play: a program that carries scan slots has no timing until the table
        supplies one, so the count is asked about one played cycle.
        """

        played = int(
            program.camera_window_count(
                self.trigger_channel,
                None if table is None else table[:1],
            )
        )
        expected = self.camera_node.request.frames_per_cycle
        if played != expected:
            raise ValueError(
                f"the template opens {played} camera windows per cycle and "
                f"this measurement reads {expected}"
            )
        if self._capture is None:
            self._capture = self.camera_node.prepare(owns_generation=False)

    def next_value(self, context: object) -> SignalValue:
        if self._capture is None:
            raise RuntimeError("the camera source was not armed")
        if context.cancel_requested():
            raise RuntimeError("the scan was cancelled")
        records = self._capture.next_cycle()
        self._taken += 1
        return SignalValue(
            CAMERA_FRAMES_OUTPUT.name,
            frames_snapshot(
                (records,),
                producer=self.camera_node.instance_id,
                generation=self._generation,
                revision=self._taken,
            ),
            MonitorCoverage(written_cells=len(records), total_cells=len(records)),
            run_record=self.camera_node.run_record,
        )

    def close(self) -> None:
        capture, self._capture = self._capture, None
        if capture is not None:
            capture.close()

    def describe(self) -> dict[str, object]:
        request = self.camera_node.request
        return {
            "source_camera": request.camera_key,
            "frames_per_cycle": request.frames_per_cycle,
        }


@dataclass(frozen=True)
class MeasurementResult:
    cycles: tuple[tuple[CameraFrameRecord, ...], ...]
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
    ) -> None:
        self.node = node
        self.camera = node.camera
        self.repeat = int(repeat)
        self.frames_per_cycle = int(frames_per_cycle)
        self.timeout = float(timeout)
        self.closed = False
        self.collected: MeasurementResult | None = None

    def collect(self, *, publish: object | None = None) -> MeasurementResult:
        """Read the armed cycles and publish them.

        ``publish`` is the host's context publisher when hosted, and None when
        the caller owns the plane directly.  Only the destination differs; the
        acquisition is one implementation.
        """

        if self.closed:
            raise RuntimeError("finite capture is closed")
        if self.collected is not None:
            return self.collected
        cycles: list[tuple[CameraFrameRecord, ...]] = []
        try:
            for _repeat in range(self.repeat):
                cycles.append(self.next_cycle())
            terminal = self.camera.finish_record_capture()
        except BaseException:
            self.camera.finish_record_capture()
            self.closed = True
            raise
        self.closed = True
        self.collected = self.node._publish_finite(tuple(cycles), terminal, publish=publish)
        return self.collected

    def next_cycle(self) -> tuple[CameraFrameRecord, ...]:
        """The next complete cycle of this capture.

        A scan takes its cycles one at a time -- that is what lets the dataset
        grow on screen while the board is still playing -- and a finite
        measurement takes them all before publishing.  Same read.
        """

        if self.closed:
            raise RuntimeError("finite capture is closed")
        records = tuple(
            self.camera.read_frame_records(
                self.frames_per_cycle,
                timeout=self.timeout,
                exact=True,
            )
        )
        if len(records) != self.frames_per_cycle:
            raise RuntimeError("camera returned an incomplete atomic cycle")
        return records

    def close(self) -> CameraCaptureTerminalRecord:
        if self.closed:
            return CameraCaptureTerminalRecord(0, True, True, True)
        self.closed = True
        return self.camera.finish_record_capture()


class MonitorCapture:
    """A repeat-zero monitor with latest-frame semantics."""

    def __init__(
        self,
        camera: CameraAdapter,
        *,
        node: "CameraMeasurementNode",
        timeout: float,
        owns_generation: bool,
        attach_live_outputs: Callable[[object], None] | None,
    ) -> None:
        self.camera = camera
        self.node = node
        self.timeout = float(timeout)
        self.owns_generation = bool(owns_generation)
        self.closed = False
        self.latest_record: CameraFrameRecord | None = None
        self._pending_records: list[CameraFrameRecord] = []
        self.slot = _CameraMonitorSlot(node)
        if self.owns_generation:
            if attach_live_outputs is not None:
                raise ValueError("a direct monitor cannot use a host live-output attachment")
            self.slot.set_change_listener(
                lambda: self.node.signal_plane.mark_changed(self.node, self.slot)
            )
            self.node.signal_plane.attach(self.node, self.slot)
        else:
            if not callable(attach_live_outputs):
                raise TypeError("a hosted monitor requires attach_live_outputs")
            attach_live_outputs(self.slot)

    def poll(self) -> CameraFrameRecord | None:
        if self.closed:
            raise RuntimeError("monitor capture is closed")
        records = self.camera.read_frame_records(1, timeout=self.timeout, exact=False)
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
            self.slot.update(tuple(pending))
            pending.clear()

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
        detached: BaseException | None = None
        if self.owns_generation:
            try:
                self.node.signal_plane.detach_live(self.node)
            except BaseException as error:  # noqa: BLE001 - the camera still goes
                detached = error
        terminal = self.camera.finish_record_capture()
        if detached is not None:
            raise detached
        return terminal


class CameraMeasurementNode:
    """One atomic camera cycle publishes one ordinary signal per frame."""

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
        self.producer = self.instance_id
        # Live frames and final publications share one counter, which continues
        # across runs so a consumer never sees this producer go backwards.
        self._generation: object | None = None
        self._revision = 0

    @property
    def request(self) -> CameraMeasurementRequest:
        return self._request

    @property
    def actual_working_point(self) -> CameraWorkingPoint | None:
        return self._actual_working_point

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
        point = self.camera.configure_measurement(
            exposure_seconds=self.request.exposure_seconds,
            roi_xywh=self.request.roi_xywh,
        )
        if not isinstance(point, CameraWorkingPoint):
            raise TypeError("camera configure_measurement must return CameraWorkingPoint")
        return point

    def _freeze_working_point(self, point: CameraWorkingPoint) -> None:
        if not isinstance(point, CameraWorkingPoint):
            raise TypeError("camera capture_working_point must return CameraWorkingPoint")
        record = {
            "node": self.instance_id,
            "parameters": {
                "exposure_seconds": self.request.exposure_seconds,
                "roi_xywh": self.request.roi_xywh,
                "repeat": self.request.repeat,
                "frames_per_cycle": self.request.frames_per_cycle,
            },
            "named_devices": {"camera": self.request.camera_key},
            "device_snapshots": {
                "camera": _camera_working_point_snapshot(point),
            },
        }
        self._actual_working_point = point
        self._run_record = record

    @property
    def run_record(self) -> dict[str, object]:
        """What this acquisition IS, frozen when the camera was armed."""

        record = self._run_record
        if record is None:
            raise RuntimeError("camera run record was not frozen after arm")
        return record

    def _next_publication_stamp(self) -> tuple[str, int]:
        generation = self._generation
        if generation is None:
            raise RuntimeError("camera publication requires an active generation")
        self._revision += 1
        return str(getattr(generation, "value", generation)), self._revision

    def prepare(self, *, owns_generation: bool = True) -> FiniteCapture:
        """Arm the camera for one acquisition.

        The frozen request is the acquisition this node performs.
        ``owns_generation`` is False when a NodeHost has already begun the
        generation for us.
        """

        if self.request.repeat <= 0:
            raise ValueError("finite prepare requires request.repeat greater than zero")
        if owns_generation:
            self._generation = self.signal_plane.begin_generation(self)
        self._configure_for_run()
        timeout = float(self.camera.timeout)
        total = self.request.repeat * self.request.frames_per_cycle
        groups = (self.request.frames_per_cycle,) * self.request.repeat
        self.camera.arm(
            total,
            source_group_sizes=groups,
            buffer_frame_count=total,
            timeout=timeout,
        )
        try:
            self._freeze_working_point(self.camera.capture_working_point())
            return FiniteCapture(
                self,
                repeat=self.request.repeat,
                frames_per_cycle=self.request.frames_per_cycle,
                timeout=timeout,
            )
        except BaseException:
            self.camera.finish_record_capture()
            raise

    def measure(self) -> MeasurementResult:
        """Collect externally triggered frames into one final publication."""

        return self.prepare().collect()

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
                attach_live_outputs=context.attach_live_outputs,
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
        capture = self.prepare(owns_generation=False)
        result = capture.collect(publish=context.publish_final)
        return {
            "cycles": len(result.cycles),
            "signals": tuple(
                self.signal_key(value.name)
                for value in self.dataset_output_declarations
            ),
        }

    def monitor(
        self,
        *,
        owns_generation: bool = True,
        attach_live_outputs: Callable[[object], None] | None = None,
    ) -> MonitorCapture:
        cycle_size = self.frames_per_cycle
        buffer_frames = 4 * cycle_size
        if self.request.repeat != 0:
            raise ValueError("monitor requires request.repeat equal to zero")
        owns_generation = bool(owns_generation)
        if owns_generation:
            if attach_live_outputs is not None:
                raise ValueError("a direct monitor cannot use a host live-output attachment")
            self._generation = self.signal_plane.begin_generation(self)
        elif not callable(attach_live_outputs):
            raise TypeError("a hosted monitor requires attach_live_outputs")
        self._configure_for_run()
        timeout = float(self.camera.timeout)
        self.camera.arm(
            None,
            source_group_sizes=None if cycle_size == 1 else (cycle_size,),
            buffer_frame_count=buffer_frames,
            timeout=timeout,
        )
        try:
            self._freeze_working_point(self.camera.capture_working_point())
            return MonitorCapture(
                self.camera,
                node=self,
                timeout=timeout,
                owns_generation=owns_generation,
                attach_live_outputs=attach_live_outputs,
            )
        except BaseException:
            self.camera.finish_record_capture()
            raise

    def _publish_finite(
        self,
        cycles: tuple[tuple[CameraFrameRecord, ...], ...],
        terminal: CameraCaptureTerminalRecord,
        *,
        publish: object | None = None,
    ) -> MeasurementResult:
        if not cycles:
            raise ValueError("camera publication requires at least one cycle")
        generation, revision = self._next_publication_stamp()
        outputs = {
            CAMERA_FRAMES_OUTPUT.name: FinalDatasetOutput(
                CAMERA_FRAMES_OUTPUT,
                frames_snapshot(
                    cycles,
                    producer=self.instance_id,
                    generation=generation,
                    revision=revision,
                ),
                self.run_record,
            )
        }
        published = publish(outputs) if publish is not None else self.signal_plane.publish_final(self, outputs)
        if not isinstance(published, dict) and not hasattr(published, "keys"):
            raise TypeError("signal_plane.publish_final must return a signal mapping")
        publication = self.signal_plane.latest_publication(
            self.signal_key(CAMERA_FRAMES_OUTPUT.name)
        )
        if not isinstance(publication, SignalPublication):
            raise RuntimeError("signal plane did not expose the final camera publication")
        return MeasurementResult(cycles, publication, terminal)


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
