"""Finite exact and live-monitor camera capture shared by the node."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from zlc_data import READOUT_EVENT, SPATIAL_X, SPATIAL_Y
from zlc_runtime import MonitorCoverage
from zlc_runtime import (
    DatasetOutputDeclaration,
    FinalDatasetOutput,
    LiveDatasetOutput,
)
from zlc_runtime import SignalPublication

from zlc_atom.devices.camera.contract import CameraAdapter, CameraCaptureTerminalRecord, CameraFrameRecord
from zlc_atom.data import snapshot_from_array
from zlc_atom.nodes._framework.descriptor import NodeKind, runtime_kind
from zlc_atom.nodes._framework.generation import ProducerRuns
from zlc_atom.nodes._framework.provenance import ProvenanceRecorder


_FRAMES_DECLARATION = DatasetOutputDeclaration("frames", "camera.frames.v1")


def _frames_array(cycles: tuple[tuple[CameraFrameRecord, ...], ...]) -> np.ndarray:
    """Materialize one finite camera publication without carrying record objects."""

    if not cycles:
        raise ValueError("camera publication requires at least one cycle")
    return np.stack(
        [np.stack([np.asarray(record.image) for record in cycle], axis=0) for cycle in cycles],
        axis=0,
    )


class _CameraMonitorSlot:
    """Application-owned live slot consumed by the runtime plane."""

    def __init__(self, node: "CameraMeasurementNode") -> None:
        self.node = node
        self.latest: CameraFrameRecord | None = None
        #: Whether the frame in ``latest`` has been handed to a consumer.  A
        #: miss is a frame overwritten BEFORE anyone read it; counting every
        #: frame that merely arrived after another one made the number equal to
        #: the frame count, so the coverage a live plot uses to say "you are
        #: behind" said it from the second frame onward, forever.
        self._delivered = True
        self.revision = 0
        self.missed_events = 0
        self.closed = False

    def update(self, record: CameraFrameRecord) -> None:
        if self.closed:
            raise RuntimeError("camera monitor slot is closed")
        if self.latest is not None and not self._delivered:
            self.missed_events += 1
        self.latest = record
        self._delivered = False
        self.revision += 1

    def freeze_live_outputs(self) -> dict[str, LiveDatasetOutput]:
        if self.closed:
            raise RuntimeError("camera monitor slot is closed")
        record = self.latest
        if record is None:
            raise RuntimeError("camera monitor slot has no accepted frame")
        array = np.asarray(record.image)[None, None, ...]
        snapshot = snapshot_from_array(
            array,
            producer=self.node.instance_id,
            signal="frames",
            roles=(READOUT_EVENT, SPATIAL_Y, SPATIAL_X),
            generation=self.node.runs.generation,
            revision=self.node.runs.next_revision(),
        )
        coverage = MonitorCoverage(
            written_cells=1,
            total_cells=1,
            missed_events=self.missed_events,
            current_gap=False,
        )
        self._delivered = True
        return {
            "frames": LiveDatasetOutput(_FRAMES_DECLARATION, snapshot, coverage),
        }

    def close(self) -> None:
        self.closed = True


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

    def __init__(self, camera: CameraAdapter, *, node: "CameraMeasurementNode", timeout: float) -> None:
        self.camera = camera
        self.node = node
        self.timeout = float(timeout)
        self.closed = False
        self.latest_record: CameraFrameRecord | None = None
        self.slot = _CameraMonitorSlot(node)
        self.node.signal_plane.attach(self.node, self.slot)

    def poll(self) -> CameraFrameRecord | None:
        if self.closed:
            raise RuntimeError("monitor capture is closed")
        records = self.camera.read_frame_records(1, timeout=self.timeout, exact=False)
        if not records:
            return None
        self.latest_record = records[-1]
        self.slot.update(self.latest_record)
        self.node.signal_plane.mark_changed(self.node, self.slot)
        return self.latest_record

    def close(self) -> CameraCaptureTerminalRecord:
        """Detach from the plane and disarm the camera -- both, always.

        The detach came first with nothing protecting the disarm, so a plane
        that refused to let go left the camera armed forever: producing frames
        into a buffer nobody would read, and refusing the next arm because the
        previous one was still running.
        """

        if self.closed:
            return CameraCaptureTerminalRecord(0, True, True, True)
        self.closed = True
        detached: BaseException | None = None
        try:
            self.node.signal_plane.detach_live(self.node)
        except BaseException as error:  # noqa: BLE001 - the camera still goes
            detached = error
        terminal = self.camera.finish_record_capture()
        if detached is not None:
            raise detached
        return terminal


class CameraMeasurementNode:
    """One atomic camera cycle yields one publication containing all frames."""

    #: Derived from the domain layer, never declared twice.
    kind = runtime_kind(NodeKind.MEASUREMENT)

    def __init__(
        self,
        *,
        camera: CameraAdapter,
        signal_plane: object,
        producer: str = "camera_measurement",
        timeout: float | None = None,
        repeat: int = 1,
        frames_per_cycle: int = 1,
    ) -> None:
        if not isinstance(camera, CameraAdapter):
            raise TypeError("camera must implement CameraAdapter")
        self.camera = camera
        if signal_plane is None:
            raise TypeError("signal_plane must be supplied by the runtime owner")
        self.signal_plane = signal_plane
        self.instance_id = str(producer).strip()
        if not self.instance_id:
            raise ValueError("producer must be non-empty")
        self.producer = self.instance_id
        self.timeout = camera.timeout if timeout is None else float(timeout)
        # One revision line per producer: live frames and the final publication
        # advance the same counter, so a consumer never sees it go backwards.
        self.runs = ProducerRuns()
        # Captured once when a run begins; constant across its shots, because a
        # parameter that changes during a run IS the scan and is already in the data.
        self.provenance = ProvenanceRecorder()
        # The acquisition this node was configured to perform.  The descriptor
        # used to freeze these and drop them, so a hosted run had nothing to
        # perform and no record could say what was asked for.
        self.repeat = int(repeat)
        self.frames_per_cycle = int(frames_per_cycle)
        if self.repeat <= 0 or self.frames_per_cycle <= 0:
            raise ValueError("repeat and frames_per_cycle must be positive")

    @property
    def dataset_output_declarations(self) -> tuple[DatasetOutputDeclaration, ...]:
        return (_FRAMES_DECLARATION,)

    def signal_key(self, output_name: str) -> str:
        if str(output_name) != "frames":
            raise KeyError(f"unknown camera output {output_name!r}")
        return f"@logic/{self.instance_id}/frames"


    def prepare(
        self,
        *,
        repeat: int | None = None,
        frames_per_cycle: int | None = None,
        owns_generation: bool = True,
    ) -> FiniteCapture:
        """Arm the camera for one acquisition.

        ``repeat`` and ``frames_per_cycle`` default to the acquisition this node
        was configured with, so a hosted run and a notebook call describe the
        same measurement.  ``owns_generation`` is False when a NodeHost has
        already begun the generation for us.
        """

        repeat = self.repeat if repeat is None else int(repeat)
        frames_per_cycle = (
            self.frames_per_cycle if frames_per_cycle is None else int(frames_per_cycle)
        )
        if repeat <= 0 or frames_per_cycle <= 0:
            raise ValueError("repeat and frames_per_cycle must be positive")
        if owns_generation:
            # A run boundary is where provenance is re-taken.  It never was:
            # capture() records only the first time and nothing called reset,
            # so every archive after the first recorded the FIRST run's
            # apparatus -- the camera settings, the pulse, all of it -- for a
            # node that exists to be run again and again.
            self.runs.begin(self.signal_plane, self)
            self.provenance.reset()
        self.provenance.capture(self)
        total = repeat * frames_per_cycle
        groups = (frames_per_cycle,) * repeat
        self.camera.arm(total, source_group_sizes=groups, buffer_frame_count=total, timeout=self.timeout)
        try:
            return FiniteCapture(
                self,
                repeat=repeat,
                frames_per_cycle=frames_per_cycle,
                timeout=self.timeout,
            )
        except BaseException:
            self.camera.finish_record_capture()
            raise

    def measure(
        self,
        *,
        repeat: int | None = None,
        frames_per_cycle: int | None = None,
    ) -> MeasurementResult:
        """Collect externally triggered frames into one final publication."""

        return self.prepare(repeat=repeat, frames_per_cycle=frames_per_cycle).collect()

    def execute(self, context: object) -> dict[str, object]:
        """Hosted entry point: the same acquisition, published through the host.

        A NodeHost has already begun the generation, and publications must go
        through its context so the host can observe them.  Everything else --
        arming, reading cycles, building the snapshot -- is the identical code
        the notebook path runs, because a second implementation is how a virtual
        bench and a real one start to disagree.
        """

        self.runs.adopt(context.generation)
        self.provenance.capture(self)
        capture = self.prepare(owns_generation=False)
        result = capture.collect(publish=context.publish_final)
        return {"cycles": len(result.cycles), "signal": self.signal_key("frames")}

    def monitor(self, *, buffer_frames: int = 1) -> MonitorCapture:
        buffer_frames = int(buffer_frames)
        if buffer_frames <= 0:
            raise ValueError("buffer_frames must be positive")
        self.runs.begin(self.signal_plane, self)
        self.provenance.reset()
        self.camera.arm(
            None,
            source_group_sizes=None,
            buffer_frame_count=buffer_frames,
            timeout=self.timeout,
        )
        return MonitorCapture(self.camera, node=self, timeout=self.timeout)

    def _publish_finite(
        self,
        cycles: tuple[tuple[CameraFrameRecord, ...], ...],
        terminal: CameraCaptureTerminalRecord,
        *,
        publish: object | None = None,
    ) -> MeasurementResult:
        snapshot = snapshot_from_array(
            _frames_array(cycles),
            producer=self.instance_id,
            signal="frames",
            roles=(READOUT_EVENT, SPATIAL_Y, SPATIAL_X),
            generation=self.runs.generation,
            revision=self.runs.next_revision(),
        )
        outputs = {"frames": FinalDatasetOutput(_FRAMES_DECLARATION, snapshot)}
        published = publish(outputs) if publish is not None else self.signal_plane.publish_final(self, outputs)
        if not isinstance(published, dict) and not hasattr(published, "keys"):
            raise TypeError("signal_plane.publish_final must return a signal mapping")
        publication = self.signal_plane.latest_publication(self.signal_key("frames"))
        if not isinstance(publication, SignalPublication):
            raise RuntimeError("signal plane did not expose the final camera publication")
        return MeasurementResult(cycles, publication, terminal)


__all__ = ["CameraMeasurementNode", "FiniteCapture", "MeasurementResult", "MonitorCapture"]
