"""Finite exact and live-monitor camera capture shared by the node."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import numpy as np
from zlc_data import SPATIAL_X, SPATIAL_Y
from zlc_runtime import MonitorCoverage
from zlc_runtime import (
    DatasetOutputDeclaration,
    FinalDatasetOutput,
    LiveDatasetOutput,
)
from zlc_runtime import SignalPublication

from zlc_atom.devices.camera.contract import (
    CameraAdapter,
    CameraCaptureTerminalRecord,
    CameraFrameRecord,
    CameraWorkingPoint,
)
from zlc_atom.data import snapshot_from_array
from zlc_atom.nodes._framework.generation import ProducerRuns
from zlc_atom.nodes._framework.provenance import ProvenanceRecorder


_CAMERA_FRAME_CONTRACT = "camera.frames.v1"


def camera_frame_output_declarations(
    frames_per_cycle: int,
) -> tuple[DatasetOutputDeclaration, ...]:
    """Declare one ordinary 2-D signal for each frame in a camera cycle."""

    return tuple(
        DatasetOutputDeclaration(f"frame_{index}", _CAMERA_FRAME_CONTRACT)
        for index in range(int(frames_per_cycle))
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
    timeout_seconds: float

    def __post_init__(self) -> None:
        camera_key = str(self.camera_key).strip()
        if not camera_key:
            raise ValueError("camera_key must be non-empty")
        exposure = float(self.exposure_seconds)
        timeout = float(self.timeout_seconds)
        if not np.isfinite(exposure) or exposure <= 0:
            raise ValueError("exposure_seconds must be positive and finite")
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
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
        object.__setattr__(self, "timeout_seconds", timeout)


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
        revision = self.node.runs.next_revision()
        coverage = MonitorCoverage(
            written_cells=1,
            total_cells=1,
        )
        return {
            declaration.name: LiveDatasetOutput(
                declaration,
                snapshot_from_array(
                    np.asarray(records[index].image)[None, ...],
                    producer=self.node.instance_id,
                    signal=declaration.name,
                    roles=(SPATIAL_Y, SPATIAL_X),
                    generation=self.node.runs.generation,
                    revision=revision,
                ),
                coverage,
                self.node._require_run_record(),
            )
            for index, declaration in enumerate(
                self.node.dataset_output_declarations
            )
        }

    def close(self) -> None:
        self.closed = True
        self._change_listener = None


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
                records = tuple(
                    self.camera.read_frame_records(
                        self.frames_per_cycle,
                        timeout=self.timeout,
                        exact=True,
                    )
                )
                if len(records) != self.frames_per_cycle:
                    raise RuntimeError("camera returned an incomplete atomic cycle")
                cycles.append(records)
            terminal = self.camera.finish_record_capture()
        except BaseException:
            self.camera.finish_record_capture()
            self.closed = True
            raise
        self.closed = True
        self.collected = self.node._publish_finite(tuple(cycles), terminal, publish=publish)
        return self.collected

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
        # One revision line per producer: live frames and the final publication
        # advance the same counter, so a consumer never sees it go backwards.
        self.runs = ProducerRuns()
        # Captured once when a run begins; constant across its shots, because a
        # parameter that changes during a run IS the scan and is already in the data.
        self.provenance = ProvenanceRecorder()

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
    def timeout_seconds(self) -> float:
        return self.request.timeout_seconds

    @property
    def dataset_output_declarations(self) -> tuple[DatasetOutputDeclaration, ...]:
        return camera_frame_output_declarations(self.frames_per_cycle)

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
        self._actual_working_point = point
        self._run_record = {
            "node": self.instance_id,
            "parameters": {
                "exposure_seconds": self.request.exposure_seconds,
                "roi_xywh": self.request.roi_xywh,
                "repeat": self.request.repeat,
                "frames_per_cycle": self.request.frames_per_cycle,
                "timeout_seconds": self.request.timeout_seconds,
            },
            "named_devices": {"camera": self.request.camera_key},
            "device_snapshots": {
                "camera": _camera_working_point_snapshot(point),
            },
        }
        return point

    def _require_run_record(self) -> dict[str, object]:
        record = self._run_record
        if record is None:
            raise RuntimeError("camera run record was not frozen after configure")
        return record

    def prepare(self, *, owns_generation: bool = True) -> FiniteCapture:
        """Arm the camera for one acquisition.

        The frozen request is the acquisition this node performs.
        ``owns_generation`` is False when a NodeHost has already begun the
        generation for us.
        """

        if self.request.repeat <= 0:
            raise ValueError("finite prepare requires request.repeat greater than zero")
        if owns_generation:
            # A run boundary is where provenance is re-taken.  It never was:
            # capture() records only the first time and nothing called reset,
            # so every archive after the first recorded the FIRST run's
            # apparatus -- the camera settings, the pulse, all of it -- for a
            # node that exists to be run again and again.
            self.runs.begin(self.signal_plane, self)
            self.provenance.reset()
        self._configure_for_run()
        self.provenance.capture(self)
        total = self.request.repeat * self.request.frames_per_cycle
        groups = (self.request.frames_per_cycle,) * self.request.repeat
        self.camera.arm(
            total,
            source_group_sizes=groups,
            buffer_frame_count=total,
            timeout=self.timeout_seconds,
        )
        try:
            return FiniteCapture(
                self,
                repeat=self.request.repeat,
                frames_per_cycle=self.request.frames_per_cycle,
                timeout=self.timeout_seconds,
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

        self.runs.adopt(context.generation)
        self.provenance.reset()
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
        buffer_frames: int | None = None,
        owns_generation: bool = True,
        attach_live_outputs: Callable[[object], None] | None = None,
    ) -> MonitorCapture:
        cycle_size = self.frames_per_cycle
        minimum_frames = 16 * cycle_size
        if buffer_frames is None:
            buffer_frames = minimum_frames
        else:
            buffer_frames = int(buffer_frames)
            if buffer_frames <= 0:
                raise ValueError("buffer_frames must be positive")
            buffer_frames = max(buffer_frames, minimum_frames)
            buffer_frames = ((buffer_frames + cycle_size - 1) // cycle_size) * cycle_size
        if self.request.repeat != 0:
            raise ValueError("monitor requires request.repeat equal to zero")
        owns_generation = bool(owns_generation)
        if owns_generation:
            if attach_live_outputs is not None:
                raise ValueError("a direct monitor cannot use a host live-output attachment")
            self.runs.begin(self.signal_plane, self)
            self.provenance.reset()
        elif not callable(attach_live_outputs):
            raise TypeError("a hosted monitor requires attach_live_outputs")
        self._configure_for_run()
        self.provenance.capture(self)
        self.camera.arm(
            None,
            source_group_sizes=(self.frames_per_cycle,),
            buffer_frame_count=buffer_frames,
            timeout=self.timeout_seconds,
        )
        try:
            return MonitorCapture(
                self.camera,
                node=self,
                timeout=self.timeout_seconds,
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
        revision = self.runs.next_revision()
        outputs = {
            declaration.name: FinalDatasetOutput(
                declaration,
                snapshot_from_array(
                    np.stack(
                        [np.asarray(cycle[index].image) for cycle in cycles],
                        axis=0,
                    ),
                    producer=self.instance_id,
                    signal=declaration.name,
                    roles=(SPATIAL_Y, SPATIAL_X),
                    generation=self.runs.generation,
                    revision=revision,
                ),
                self._require_run_record(),
            )
            for index, declaration in enumerate(self.dataset_output_declarations)
        }
        published = publish(outputs) if publish is not None else self.signal_plane.publish_final(self, outputs)
        if not isinstance(published, dict) and not hasattr(published, "keys"):
            raise TypeError("signal_plane.publish_final must return a signal mapping")
        publication = self.signal_plane.latest_publication(self.signal_key("frame_0"))
        if not isinstance(publication, SignalPublication):
            raise RuntimeError("signal plane did not expose the final camera publication")
        return MeasurementResult(cycles, publication, terminal)


__all__ = [
    "CameraMeasurementNode",
    "CameraMeasurementRequest",
    "FiniteCapture",
    "MeasurementResult",
    "MonitorCapture",
    "camera_frame_output_declarations",
]
