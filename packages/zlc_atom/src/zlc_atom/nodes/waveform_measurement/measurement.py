"""Ordered native waveform records, independent of read and display cadence.

Adapters own bounded receive queues. Runtime alone retains run data/history.
Each native record is published in order, preserving its samples and timestamp.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from math import ceil
from time import monotonic, time_ns

import numpy as np
from zlc_data import COMPONENT, READOUT_EVENT, SAMPLE_TIME, AxisSpec, DomainSpec, OwnedSnapshot
from zlc_runtime import (
    DatasetCoverage,
    DatasetOutputDeclaration,
    LiveDatasetOutput,
    MonitorCoverage,
    SignalValue,
)

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.data import cell_axis_id, snapshot_from_array
from zlc_atom.devices.waveform.contract import (
    WaveformAcquisitionMode,
    WaveformCaptureTerminalRecord,
    WaveformOutput,
    WaveformRecord,
    WaveformSource,
    WaveformWorkingPoint,
    validate_waveform_outputs,
)
from zlc_atom.nodes._framework.descriptor import NodePreviewSpec


#: How often a capture comes back to see whether it has been asked to stop;
#: the read length is the cancel latency, and it belongs to the loop.
_CANCEL_RESPONSE_SECONDS = 0.05

WAVEFORM_MEASUREMENT_SCHEMA = AuthoringSchema((
    AuthoringField("repeat", "int", "Records (0 = continuous)", 0, minimum=0),
    AuthoringField("buffer_seconds", "float", "Receive buffer", 2.0, minimum=0.01, unit="s"),
))


def waveform_outputs(
    outputs: tuple[WaveformOutput, ...],
) -> tuple[DatasetOutputDeclaration, ...]:
    """One signal per quantity a source carries, each with a shot history."""

    return tuple(
        DatasetOutputDeclaration(
            output.name, f"waveform.{output.name}", index_by_source=True
        )
        for output in validate_waveform_outputs(outputs)
    )


def waveform_preview(source: WaveformSource) -> NodePreviewSpec:
    """The first quantity, as a rolling trace of shots or as the trace one shot is."""

    (first, *_rest) = waveform_outputs(source.outputs)
    return NodePreviewSpec(first, "rolling" if int(source.record_samples) == 1 else "curve")


@lru_cache(maxsize=64)
def _channel_axis(producer: str, signal: str, labels: tuple[str, ...]) -> AxisSpec:
    return AxisSpec(
        cell_axis_id(producer, signal, 0, COMPONENT),
        "channel",
        COMPONENT,
        len(labels),
        coordinate_labels=labels,
    )


@lru_cache(maxsize=64)
def _sample_axis(producer: str, signal: str, interval: float, samples: int, continuous: bool) -> AxisSpec:
    role = SAMPLE_TIME if continuous else READOUT_EVENT
    return AxisSpec(cell_axis_id(producer, signal, 1, role),
        "sample time" if continuous else "record time", role, int(samples),
        coordinates=tuple(index * interval for index in range(int(samples))), unit="s")


def shot_snapshot(
    record: WaveformRecord,
    *,
    output: WaveformOutput,
    producer: str,
    generation: object,
    revision: int,
    sample_interval_seconds: float,
    continuous: bool = False,
) -> OwnedSnapshot:
    """Continuous samples are Point rows; triggered samples stay in the cell."""

    picked = record.samples[:, list(output.columns)]
    labels = output.channel_labels
    point_axes = ()
    if picked.shape[0] == 1:
        values = np.ascontiguousarray(picked[0])[None]
        cell_axes: tuple[AxisSpec, ...] = (_channel_axis(producer, output.name, labels),)
    elif continuous:
        values = np.ascontiguousarray(picked)[None]
        cell_axes = (_channel_axis(producer, output.name, labels),)
        point_axes = (_sample_axis(producer, output.name, sample_interval_seconds, picked.shape[0], True),)
    else:
        values = np.ascontiguousarray(picked.T)[None]
        cell_axes = (
            _channel_axis(producer, output.name, labels),
            _sample_axis(producer, output.name, sample_interval_seconds, picked.shape[0], continuous),
        )
    return snapshot_from_array(
        values,
        producer=producer,
        signal=output.name,
        point_axes=point_axes,
        cell_axes=cell_axes,
        value_unit=output.unit,
        generation=str(getattr(generation, "value", generation)),
        revision=int(revision),
    )


def _record_seconds(node: "WaveformMeasurementNode") -> float:
    return int(node.sampler.record_samples) * node.working_point.sample_interval_seconds


def _stopped_terminal(
    terminal: WaveformCaptureTerminalRecord,
) -> WaveformCaptureTerminalRecord:
    """The device's terminal, proving the capture stopped and joined.

    The source stops before the caller drains its already accepted records.
    """

    if not (terminal.source_stopped and terminal.joined):
        raise RuntimeError(
            "waveform terminal evidence is incomplete: the capture did not stop and join"
        )
    return terminal


def _working_point_snapshot(
    source: WaveformSource, point: WaveformWorkingPoint
) -> dict[str, object]:
    return {
        "acquisition_mode": str(point.acquisition_mode),
        "time_basis": point.time_basis,
        "sample_interval_seconds": float(point.sample_interval_seconds),
        "record_samples": int(source.record_samples),
        "outputs": [
            {
                "name": output.name,
                "unit": output.unit,
                "channels": list(output.channel_labels),
            }
            for output in source.outputs
        ],
        "settings": dict(point.settings),
    }


@dataclass(frozen=True)
class WaveformMeasurementRequest:
    """Native record count and receive capacity, not a polling clock."""

    sampler_key: str
    repeat: int
    buffer_seconds: float = 2.0

    def __post_init__(self) -> None:
        key = str(self.sampler_key).strip()
        if not key:
            raise ValueError("sampler_key must be non-empty")
        if isinstance(self.repeat, bool) or int(self.repeat) != self.repeat or self.repeat < 0:
            raise ValueError("repeat must be a non-negative integer")
        seconds = float(self.buffer_seconds)
        if not np.isfinite(seconds) or seconds < 0.01:
            raise ValueError("buffer_seconds must be finite and at least 0.01 s")
        object.__setattr__(self, "sampler_key", key)
        object.__setattr__(self, "repeat", int(self.repeat))
        object.__setattr__(self, "buffer_seconds", seconds)


class FiniteCapture:
    """An armed finite capture: ``repeat`` shots, each one reading."""

    def __init__(
        self,
        node: "WaveformMeasurementNode",
        *,
        owns_generation: bool,
        should_stop: Callable[[], bool] | None,
    ) -> None:
        self.node = node
        self.owns_generation = bool(owns_generation)
        self.should_stop = should_stop
        self.closed = False
        self.completed_shots = 0
        self.terminal: WaveformCaptureTerminalRecord | None = None

    def collect(
        self,
        *,
        commit_shot: Callable[[WaveformRecord, int], None] | None = None,
    ) -> int:
        """Read every shot and publish it; answers how many shots were kept."""

        if self.closed:
            raise RuntimeError("finite capture is closed")
        if commit_shot is None:
            if not self.owns_generation:
                raise TypeError("hosted finite capture requires commit_shot")
            commit_shot = self.node._commit_direct_shot
        try:
            for index in range(self.node.repeat):
                record = self.next_shot()
                if record is None:
                    break
                commit_shot(record, index)
            terminal = self.close()
            # Stop freezes the receive count before draining. No subsequent
            # read can pull a sample acquired after this capture stopped.
            for index in range(self.completed_shots, min(self.node.repeat, terminal.produced_count)):
                record, = self.node.sampler.read_records(1, timeout=0.0, exact=True)
                commit_shot(record, index)
                self.completed_shots += 1
            self.terminal = replace(terminal, no_more_records=self.completed_shots == terminal.produced_count)
        except BaseException as error:
            # The source is stopped whatever happened, and a source that
            # failed raises again from its terminal; the generation is let
            # go either way, or the plane would hold a run nobody finishes.
            try:
                if not self.closed:
                    self.closed = True
                    self.node.sampler.finish_record_capture()
            except BaseException as cleanup_error:
                error.add_note(f"waveform cleanup also failed: {cleanup_error}")
            finally:
                if self.owns_generation:
                    self.node.signal_plane.retire(self.node)
            raise
        if self.owns_generation:
            if self.completed_shots:
                self.node.signal_plane.seal_committed(
                    self.node, cut_short=self.completed_shots < self.node.repeat
                )
            else:
                self.node.signal_plane.retire(self.node)
        return self.completed_shots

    def next_shot(self) -> WaveformRecord | None:
        """Wait for the next native record; never substitute a newer one."""
        if self.closed:
            raise RuntimeError("finite capture is closed")
        node = self.node
        deadline = monotonic() + _record_seconds(node) + node.sampler.timeout
        while self.should_stop is None or not self.should_stop():
            records = node.sampler.read_records(1, timeout=_CANCEL_RESPONSE_SECONDS, exact=False)
            if records:
                self.completed_shots += 1
                return records[0]
            if monotonic() >= deadline:
                raise TimeoutError("waveform source stopped delivering records")
        return None

    def close(self) -> WaveformCaptureTerminalRecord:
        if self.terminal is not None:
            return self.terminal
        if self.closed:
            raise RuntimeError("finite capture closed without terminal evidence")
        terminal = _stopped_terminal(self.node.sampler.finish_record_capture())
        self.closed = True
        self.terminal = terminal
        return terminal


class MonitorCapture:
    """A repeat-zero monitor: every shot replaces the last on the plane."""

    def __init__(
        self,
        node: "WaveformMeasurementNode",
        *,
        owns_generation: bool,
        commit_live: Callable[..., Mapping[str, SignalValue]] | None,
    ) -> None:
        self.node = node
        self.owns_generation = bool(owns_generation)
        self.closed = False
        self.terminal: WaveformCaptureTerminalRecord | None = None
        self._revision = 0
        self._deadline = monotonic() + _record_seconds(node) + node.sampler.timeout
        if self.owns_generation:
            if commit_live is not None:
                raise ValueError("a direct monitor cannot use a host commit function")
            self._commit_live = self.node._commit_direct_outputs
        else:
            if not callable(commit_live):
                raise TypeError("a hosted monitor requires commit_live")
            self._commit_live = commit_live

    @property
    def revision(self) -> int:
        return self._revision

    def poll(self) -> int:
        """Consume the next record, independently of display cadence."""
        if self.closed:
            raise RuntimeError("monitor capture is closed")
        records = self.node.sampler.read_records(1, timeout=_CANCEL_RESPONSE_SECONDS, exact=False)
        if not records:
            if monotonic() >= self._deadline:
                raise TimeoutError("waveform source stopped delivering records")
            return 0
        self._publish(records[0])
        self._deadline = monotonic() + _record_seconds(self.node) + self.node.sampler.timeout
        return 1

    def _publish(self, record: WaveformRecord) -> None:
        self._revision += 1
        self._commit_live(self.node._shot_outputs(record, revision=self._revision))

    def close(self, *, drain: bool = True) -> WaveformCaptureTerminalRecord:
        if self.terminal is not None:
            return self.terminal
        self.closed = True
        try:
            terminal = _stopped_terminal(self.node.sampler.finish_record_capture())
            if drain:
                for _ in range(self._revision, terminal.produced_count):
                    record, = self.node.sampler.read_records(1, timeout=0.0, exact=True)
                    self._publish(record)
                terminal = replace(terminal, no_more_records=True)
        except BaseException:
            # A source that failed raises from its terminal; the generation
            # is let go, not left open on the plane.
            if self.owns_generation:
                self.node.signal_plane.retire(self.node)
            raise
        self.terminal = terminal
        if self.owns_generation:
            if drain and self._revision:
                self.node.signal_plane.seal_committed(self.node)
            else:
                self.node.signal_plane.retire(self.node)
        return terminal


class WaveformMeasurementNode:
    """Commit each shot of a waveform source to its declared signals."""

    def __init__(
        self,
        *,
        sampler: WaveformSource,
        request: WaveformMeasurementRequest,
        signal_plane: object,
        producer: str,
    ) -> None:
        if not isinstance(sampler, WaveformSource):
            raise TypeError("sampler must implement WaveformSource")
        if not isinstance(request, WaveformMeasurementRequest):
            raise TypeError("request must be WaveformMeasurementRequest")
        if signal_plane is None:
            raise TypeError("signal_plane must be supplied by the runtime owner")
        self.sampler = sampler
        self._request = request
        self.signal_plane = signal_plane
        self._outputs = waveform_outputs(sampler.outputs)
        self._quantities = {output.name: output for output in sampler.outputs}
        self.instance_id = str(producer).strip()
        if not self.instance_id:
            raise ValueError("producer must be non-empty")
        self._generation: object | None = None
        self._working_point: WaveformWorkingPoint | None = None
        self._run_record: dict[str, object] | None = None
        self._time_origin: float | None = None
        self._last_time: float | None = None
        self._next_ordinal = 0

    @property
    def request(self) -> WaveformMeasurementRequest:
        return self._request

    @property
    def repeat(self) -> int:
        return self._request.repeat

    @property
    def sampler_key(self) -> str:
        return self._request.sampler_key

    @property
    def generation(self) -> object:
        if self._generation is None:
            raise RuntimeError("waveform acquisition has no active generation")
        return self._generation

    @property
    def working_point(self) -> WaveformWorkingPoint:
        point = self._working_point
        if point is None:
            raise RuntimeError("waveform working point is not frozen")
        return point

    @property
    def run_record(self) -> dict[str, object]:
        record = self._run_record
        if record is None:
            raise RuntimeError("waveform run record was not frozen")
        return record

    @property
    def dataset_output_declarations(self) -> tuple[DatasetOutputDeclaration, ...]:
        return self._outputs

    def signal_key(self, output_name: str) -> str:
        name = str(output_name)
        if name not in {declaration.name for declaration in self._outputs}:
            raise KeyError(f"unknown waveform output {output_name!r}")
        return f"@logic/{self.instance_id}/{name}"

    # ------------------------------------------------------------ freezing
    def _configure(self) -> WaveformWorkingPoint:
        """Freeze how the source samples for this run."""

        point = self.sampler.working_point()
        if not isinstance(point, WaveformWorkingPoint):
            raise TypeError("waveform source working_point must return WaveformWorkingPoint")
        self._working_point = point
        self._time_origin = None
        self._last_time = None
        self._next_ordinal = 0
        self._run_record = {
            "node": self.instance_id,
            "parameters": {
                "repeat": self.repeat,
                "buffer_seconds": self.request.buffer_seconds,
            },
            "named_devices": {"sampler": self.sampler_key},
            "device_snapshots": {"sampler": _working_point_snapshot(self.sampler, point)},
            # Host call time, not a device-clock to UTC calibration.
            "started_at_ns": time_ns(),
        }
        return point

    def _shot_time(self, record: WaveformRecord) -> float:
        if record.source_ordinal != self._next_ordinal:
            raise RuntimeError(f"waveform record gap: expected {self._next_ordinal}, got {record.source_ordinal}")
        if record.samples.shape[0] != self.sampler.record_samples:
            raise ValueError("waveform record sample count changed during capture")
        if self._last_time is not None and record.time_seconds <= self._last_time:
            raise RuntimeError("waveform record clock did not advance")
        if self._time_origin is None:
            self._time_origin = record.time_seconds
        self._last_time = record.time_seconds
        self._next_ordinal += 1
        return record.time_seconds - self._time_origin

    def _shot_outputs(
        self,
        record: WaveformRecord,
        *,
        revision: int,
        index: int | None = None,
    ) -> dict[str, LiveDatasetOutput]:
        point = self.working_point
        outputs: dict[str, LiveDatasetOutput] = {}
        shot_time = self._shot_time(record)
        evidence = {"record_timing": {self.instance_id: {str(record.source_ordinal): {
            "record_time_seconds": shot_time, "source_time_seconds": record.time_seconds,
            "time_basis": point.time_basis, "host_received_at_ns": record.host_received_at_ns,
        }}}}
        for declaration in self._outputs:
            snapshot = shot_snapshot(
                record,
                output=self._quantities[declaration.name],
                producer=self.instance_id,
                generation=self.generation,
                revision=revision,
                sample_interval_seconds=point.sample_interval_seconds,
                continuous=point.acquisition_mode == WaveformAcquisitionMode.FREE_RUNNING.value,
            )
            if index is None:
                outputs[declaration.name] = LiveDatasetOutput(
                    declaration,
                    snapshot,
                    MonitorCoverage(snapshot.block.schema.point_domain.size, snapshot.block.schema.point_domain.size),
                    shot_time_seconds=shot_time,
                    event_record=evidence,
                )
                continue
            schema = snapshot.block.schema
            (repeat_axis,) = schema.repeat_domain.axes
            canonical = replace(
                schema,
                repeat_domain=DomainSpec(
                    (self.repeat,),
                    (replace(repeat_axis, size=self.repeat),),
                    (tuple(range(self.repeat)),),
                ),
            )
            outputs[declaration.name] = LiveDatasetOutput(
                declaration,
                snapshot,
                DatasetCoverage((index + 1) * schema.point_domain.size, self.repeat * schema.point_domain.size),
                canonical,
                (index, 0),
                event_record=evidence,
            )
        return outputs

    def _commit_direct_outputs(
        self, outputs: Mapping[str, LiveDatasetOutput]
    ) -> Mapping[str, SignalValue]:
        return self.signal_plane.commit_live(self, outputs)

    def _commit_direct_shot(self, record: WaveformRecord, index: int) -> None:
        self._commit_direct_outputs(
            self._shot_outputs(record, revision=index + 1, index=index)
        )

    def _arm(self) -> None:
        # Capacity absorbs stalls; it never sets sample/record/plot cadence.
        capacity = max(2, ceil(self.request.buffer_seconds / _record_seconds(self)))
        self.sampler.arm(self.repeat or None, buffer_record_count=capacity)
        # Hardware may quantize the requested rate at arm (notably DAQ).
        # Freeze its accepted clock before publishing even the first record.
        self._working_point = self.sampler.working_point()
        self._run_record["device_snapshots"]["sampler"] = _working_point_snapshot(
            self.sampler, self._working_point
        )

    # ------------------------------------------------------------- capture
    def prepare(
        self,
        *,
        owns_generation: bool = True,
        should_stop: Callable[[], bool] | None = None,
    ) -> FiniteCapture:
        if self.repeat <= 0:
            raise ValueError("finite prepare requires request.repeat greater than zero")
        owns_generation = bool(owns_generation)
        if owns_generation:
            self._generation = self.signal_plane.begin_generation(self)
        try:
            self._configure()
            self._arm()
            if owns_generation:
                self.signal_plane.set_run_record(self, self.run_record)
            return FiniteCapture(
                self, owns_generation=owns_generation, should_stop=should_stop
            )
        except BaseException:
            self.sampler.finish_record_capture()
            if owns_generation:
                self.signal_plane.retire(self)
            raise

    def monitor(
        self,
        *,
        owns_generation: bool = True,
        commit_live: Callable[..., Mapping[str, SignalValue]] | None = None,
    ) -> MonitorCapture:
        if self.repeat != 0:
            raise ValueError("monitor requires request.repeat equal to zero")
        owns_generation = bool(owns_generation)
        if owns_generation:
            if commit_live is not None:
                raise ValueError("a direct monitor cannot use a host commit function")
            self._generation = self.signal_plane.begin_generation(self)
        elif not callable(commit_live):
            raise TypeError("a hosted monitor requires commit_live")
        try:
            self._configure()
            self._arm()
            if owns_generation:
                self.signal_plane.set_run_record(self, self.run_record)
            return MonitorCapture(
                self, owns_generation=owns_generation, commit_live=commit_live
            )
        except BaseException:
            self.sampler.finish_record_capture()
            if owns_generation:
                self.signal_plane.retire(self)
            raise

    def execute(self, context: object) -> dict[str, object]:
        """Hosted entry point: the same capture, published through the host."""

        self._generation = context.generation
        signals = tuple(self.signal_key(value.name) for value in self._outputs)
        if self.repeat == 0:
            capture = self.monitor(owns_generation=False, commit_live=context.commit_live)
            try:
                context.set_run_record(self.run_record)
                context.report_ready()
                while not context.cancel_requested():
                    capture.poll()
            except BaseException as error:
                try:
                    capture.close(drain=False)
                except BaseException as cleanup_error:
                    error.add_note(f"waveform cleanup also failed: {cleanup_error}")
                raise
            capture.close()
            return {"signals": signals}
        capture = self.prepare(owns_generation=False, should_stop=context.cancel_requested)
        try:
            context.set_run_record(self.run_record)
            context.report_ready()
        except BaseException:
            capture.close()
            raise

        def commit_shot(record: WaveformRecord, index: int) -> None:
            context.commit_live(self._shot_outputs(record, revision=index + 1, index=index))
            context.report_progress("Capturing", current=index + 1, total=int(self.repeat))

        completed = capture.collect(commit_shot=commit_shot)
        return {"shots": completed, "signals": signals}


__all__ = [
    "WAVEFORM_MEASUREMENT_SCHEMA",
    "FiniteCapture",
    "MonitorCapture",
    "WaveformMeasurementNode",
    "WaveformMeasurementRequest",
    "shot_snapshot",
    "waveform_outputs",
    "waveform_preview",
]
