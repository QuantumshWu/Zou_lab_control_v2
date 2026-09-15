"""Finite exact and live-monitor capture of a waveform source: one record, one shot.

One measurement for every waveform source, as there is one camera
measurement for every camera.  What the source carries is the source's
own statement (``WaveformSource.outputs``): an IMU packet is four
quantities, a scope acquisition is one, and the measurement publishes
one signal per quantity, named by the instrument's vocabulary.  The
capture, the read cadence and the commit are the same for all of them.

Every record the measurement takes is one shot, published the moment it
is read: an IMU packet becomes ``(1) x () x (channel)`` per quantity, a
scope acquisition ``(1) x () x (channel, time)`` with the samples on a
READOUT_EVENT axis whose coordinates are seconds from the trigger.  The
history of shots is the Runtime's: every output declares
``index_by_source``, so a Rolling panel leases a window and the plane
keeps the last N shots by their own sequence.

What the measurement is told, besides how many shots, is HOW OFTEN TO READ
THE HARDWARE: at each due time the newest record is the reading and
whatever arrived in between is not published -- a magnetometer streaming
at 400 Hz read every 100 ms is ten shots a second, each the field right
then.  An interval shorter than the source's own record period reads
every record, because each due time then waits for the next one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from math import ceil
from time import monotonic, sleep, time_ns

import numpy as np
from zlc_data import COMPONENT, READOUT_EVENT, AxisSpec, DomainSpec, OwnedSnapshot
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

#: The fastest the hardware may be read: a shot every millisecond.  A shot
#: is a commit on the plane and a wake of every panel that follows it, and
#: one process publishes a few thousand of them a second before it has no
#: time left to read; the ceiling is the measurement's, stated where the
#: operator sets the cadence, so the form refuses what the run could not
#: keep up with instead of dropping records silently.
MIN_READ_INTERVAL_SECONDS = 0.001

WAVEFORM_MEASUREMENT_SCHEMA = AuthoringSchema(
    (
        AuthoringField("repeat", "int", "Repeat", 0, minimum=0),
        AuthoringField(
            "read_interval_seconds",
            "float",
            "Read interval",
            0.01,
            minimum=MIN_READ_INTERVAL_SECONDS,
            unit="s",
        ),
    )
)


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
def _sample_axis(producer: str, signal: str, interval: float, samples: int) -> AxisSpec:
    """The time axis inside one acquisition, built once per working point.

    Coordinates are seconds from the record's own first sample, so every
    shot of a run shares one axis object and the schema cache hits.
    """

    return AxisSpec(
        cell_axis_id(producer, signal, 1, READOUT_EVENT),
        "time",
        READOUT_EVENT,
        int(samples),
        coordinates=tuple(index * interval for index in range(int(samples))),
        unit="s",
    )


def shot_snapshot(
    record: WaveformRecord,
    *,
    output: WaveformOutput,
    producer: str,
    generation: object,
    revision: int,
    sample_interval_seconds: float,
) -> OwnedSnapshot:
    """One record of one output as a dataset: (1) x () x (channel[, time])."""

    picked = record.samples[:, list(output.columns)]
    labels = output.channel_labels
    if picked.shape[0] == 1:
        values = np.ascontiguousarray(picked[0])[None]
        cell_axes: tuple[AxisSpec, ...] = (_channel_axis(producer, output.name, labels),)
    else:
        values = np.ascontiguousarray(picked.T)[None]
        cell_axes = (
            _channel_axis(producer, output.name, labels),
            _sample_axis(producer, output.name, sample_interval_seconds, picked.shape[0]),
        )
    return snapshot_from_array(
        values,
        producer=producer,
        signal=output.name,
        cell_axes=cell_axes,
        value_unit=output.unit,
        generation=str(getattr(generation, "value", generation)),
        revision=int(revision),
    )


def _newest_at_due(node: "WaveformMeasurementNode", due: float) -> WaveformRecord | None:
    """The newest record once ``due`` has passed; before it, nothing.

    Until the due time the source is only drained -- those records are not
    the reading -- and the time to the due is slept, a cancel slice at
    most.  It is slept, not waited on the source: a timed lock wait has
    the OS timer tick as its resolution (15 ms on Windows) while the sleep
    keeps well under a millisecond, and the cadence is the reading.  At
    the due time the first record is waited for (a source may not have
    produced one since the last drain) and everything behind it is taken,
    so the reading is the newest record there is, not the oldest.
    """

    remaining = due - monotonic()
    if remaining > 0.0:
        node.sampler.read_records(node.read_batch, timeout=0.0, exact=False)
        sleep(min(_CANCEL_RESPONSE_SECONDS, remaining))
        return None
    arrived = node.sampler.read_records(1, timeout=_CANCEL_RESPONSE_SECONDS, exact=False)
    if not arrived:
        return None
    newest = arrived[-1]
    while True:
        behind = node.sampler.read_records(node.read_batch, timeout=0.0, exact=False)
        if not behind:
            return newest
        newest = behind[-1]


def _due_after(due: float, interval: float) -> float:
    """The next due on the cadence grid.

    A shot that ran late stays on the grid, so the average cadence is the
    interval exactly; only a grid a whole interval or more behind re-anchors
    at now, which is a source or a node that cannot keep the cadence, and
    then the shots simply come as fast as they can.
    """

    due += interval
    now = monotonic()
    return now if due <= now - interval else due


def _stopped_terminal(
    terminal: WaveformCaptureTerminalRecord,
) -> WaveformCaptureTerminalRecord:
    """The device's terminal, proving the capture stopped and joined.

    The source's record count says nothing about the shots: a capture took
    the newest record at each due time and let the rest go, and what the
    source still holds unread is what the cadence let go.
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
    """One frozen selection: which source, how many shots, how often to read."""

    sampler_key: str
    repeat: int
    read_interval_seconds: float

    def __post_init__(self) -> None:
        key = str(self.sampler_key).strip()
        if not key:
            raise ValueError("sampler_key must be non-empty")
        if int(self.repeat) < 0:
            raise ValueError("repeat must be non-negative")
        interval = float(self.read_interval_seconds)
        if not np.isfinite(interval) or interval < MIN_READ_INTERVAL_SECONDS:
            raise ValueError(
                "read_interval_seconds must be finite and at least "
                f"{MIN_READ_INTERVAL_SECONDS:g} s"
            )
        object.__setattr__(self, "sampler_key", key)
        object.__setattr__(self, "repeat", int(self.repeat))
        object.__setattr__(self, "read_interval_seconds", interval)


class FiniteCapture:
    """An armed finite capture: ``repeat`` shots, each one record."""

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
        self.stopped = False
        self.terminal: WaveformCaptureTerminalRecord | None = None
        self._next_due = monotonic()

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
            self.close()
        except BaseException:
            # The source is stopped whatever happened, and a source that
            # failed raises again from its terminal; the generation is let
            # go either way, or the plane would hold a run nobody finishes.
            try:
                if not self.closed:
                    self.closed = True
                    self.node.sampler.finish_record_capture()
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
        """The next shot, or None if asked to stop.

        The newest record at the due time; what arrived before it is let
        go.  A source that has produced nothing by the due time is waited
        for, up to its own timeout.
        """

        if self.closed:
            raise RuntimeError("finite capture is closed")
        node = self.node
        timeout = float(node.sampler.timeout)
        # The source's timeout is how long it may take to deliver a record
        # once one is due, so it counts from the due time: a cadence longer
        # than the timeout is a slow reading, not a silent source.
        deadline = max(monotonic(), self._next_due) + timeout
        while True:
            if self.should_stop is not None and self.should_stop():
                self.stopped = True
                return None
            record = _newest_at_due(node, self._next_due)
            if record is not None:
                break
            if monotonic() >= deadline:
                raise RuntimeError(
                    f"the waveform source delivered no record within its {timeout:g} s timeout"
                )
        self._next_due = _due_after(self._next_due, node.read_interval_seconds)
        self.completed_shots += 1
        return record

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
        self._next_due = monotonic()
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
        """Take what has arrived and publish what the cadence says is a shot."""

        if self.closed:
            raise RuntimeError("monitor capture is closed")
        node = self.node
        newest = _newest_at_due(node, self._next_due)
        if newest is None:
            return 0
        self._publish(newest)
        self._next_due = _due_after(self._next_due, node.read_interval_seconds)
        return 1

    def _publish(self, record: WaveformRecord) -> None:
        self._revision += 1
        self._commit_live(self.node._shot_outputs(record, revision=self._revision))

    def close(self) -> WaveformCaptureTerminalRecord:
        if self.terminal is not None:
            return self.terminal
        self.closed = True
        try:
            terminal = self.node.sampler.finish_record_capture()
        except BaseException:
            # A source that failed raises from its terminal; the generation
            # is let go, not left open on the plane.
            if self.owns_generation:
                self.node.signal_plane.retire(self.node)
            raise
        self.terminal = terminal
        if self.owns_generation:
            if self._revision:
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

    @property
    def request(self) -> WaveformMeasurementRequest:
        return self._request

    @property
    def repeat(self) -> int:
        return self._request.repeat

    @property
    def read_interval_seconds(self) -> float:
        return self._request.read_interval_seconds

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
    def read_batch(self) -> int:
        """Records one read may take: everything a cancel slice can bring."""

        per_slice = _CANCEL_RESPONSE_SECONDS / (
            int(self.sampler.record_samples) * self.working_point.sample_interval_seconds
        )
        return max(4, int(ceil(per_slice)) + 1)

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
        self._run_record = {
            "node": self.instance_id,
            "parameters": {
                "repeat": self.repeat,
                "read_interval_seconds": self.read_interval_seconds,
            },
            "named_devices": {"sampler": self.sampler_key},
            "device_snapshots": {"sampler": _working_point_snapshot(self.sampler, point)},
            # Host epoch time at arm: the shots' times are seconds from the
            # first shot, and this is what puts that first shot on a clock.
            "started_at_ns": time_ns(),
        }
        return point

    def _shot_time(self, record: WaveformRecord) -> float:
        """Seconds from the run's first shot, on the source's own clock."""

        if self._time_origin is None:
            self._time_origin = record.time_seconds
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
        for declaration in self._outputs:
            snapshot = shot_snapshot(
                record,
                output=self._quantities[declaration.name],
                producer=self.instance_id,
                generation=self.generation,
                revision=revision,
                sample_interval_seconds=point.sample_interval_seconds,
            )
            if index is None:
                outputs[declaration.name] = LiveDatasetOutput(
                    declaration,
                    snapshot,
                    MonitorCoverage(1, 1),
                    self.run_record,
                    shot_time_seconds=shot_time,
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
                DatasetCoverage(index + 1, self.repeat),
                self.run_record,
                canonical,
                (index, 0),
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
        """Arm for sampling: the source runs until the capture stops it.

        A read takes the newest record at each due time and drains the
        rest, so the buffer only has to hold a few slices' worth; a source
        that outruns it drops the oldest, which the read would have let go
        anyway.
        """

        self.sampler.arm(None, buffer_record_count=4 * self.read_batch)

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
                context.report_ready()
                while not context.cancel_requested():
                    capture.poll()
            finally:
                capture.close()
            return {"signals": signals}
        capture = self.prepare(owns_generation=False, should_stop=context.cancel_requested)
        try:
            context.report_ready()
        except BaseException:
            capture.stopped = True
            capture.close()
            raise

        def commit_shot(record: WaveformRecord, index: int) -> None:
            context.commit_live(self._shot_outputs(record, revision=index + 1, index=index))
            context.report_progress("Capturing", current=index + 1, total=int(self.repeat))

        completed = capture.collect(commit_shot=commit_shot)
        return {"shots": completed, "signals": signals}


__all__ = [
    "MIN_READ_INTERVAL_SECONDS",
    "WAVEFORM_MEASUREMENT_SCHEMA",
    "FiniteCapture",
    "MonitorCapture",
    "WaveformMeasurementNode",
    "WaveformMeasurementRequest",
    "shot_snapshot",
    "waveform_outputs",
    "waveform_preview",
]
