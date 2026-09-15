"""Finite exact and live-monitor capture of a waveform source.

This package is a LIBRARY, not a node: it carries no ``logic_node.py``.
Two nodes stand on it -- the IMU measurement and the scope measurement --
and what differs between them is only what they publish: an IMU packet is
four quantities, a scope acquisition is one.  The capture itself, the
event grouping, the ordinal discipline and the commit are the same for
both, and live here once.

An event is ``records_per_event`` consecutive records concatenated along
time and published as ``(repeat) x () x (channel, time)``: the channels are
a labelled COMPONENT axis, the samples a READOUT_EVENT axis whose
coordinates are seconds from the event's first sample.  Time lives in the
cell domain and not the point domain on purpose: a point axis is
multiplied by every scan a signal is composed into, and a thousand samples
times a hundred scan points is a hundred thousand Python codes per commit;
a cell axis stays one dense dimension whatever wraps it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from time import monotonic

import numpy as np
from zlc_data import COMPONENT, READOUT_EVENT, AxisSpec, DomainSpec, OwnedSnapshot
from zlc_runtime import (
    DatasetCoverage,
    DatasetOutputDeclaration,
    LiveDatasetOutput,
    MonitorCoverage,
    SignalPublication,
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
)


#: How often a capture comes back to see whether it has been asked to stop;
#: the read length is the cancel latency, and it belongs to the loop.
_CANCEL_RESPONSE_SECONDS = 0.05


def waveform_authoring_schema(*, records_per_event: int) -> AuthoringSchema:
    """The two things a waveform measurement is told: how many events, how big."""

    return AuthoringSchema(
        (
            AuthoringField("repeat", "int", "Repeat", 0, minimum=0),
            AuthoringField(
                "records_per_event",
                "int",
                "Records per event",
                int(records_per_event),
                minimum=1,
            ),
        )
    )


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
    """The time axis of one event, built once per working point.

    Coordinates are seconds from the event's own first sample, so every
    event of a run shares one axis object and the schema cache hits.
    """

    return AxisSpec(
        cell_axis_id(producer, signal, 1, READOUT_EVENT),
        "time",
        READOUT_EVENT,
        int(samples),
        coordinates=tuple(index * interval for index in range(int(samples))),
        unit="s",
    )


def event_snapshot(
    records: Sequence[WaveformRecord],
    *,
    output: WaveformOutput,
    producer: str,
    generation: object,
    revision: int,
    sample_interval_seconds: float,
) -> OwnedSnapshot:
    """One event of one output as a dataset: (1) x () x (channel, time)."""

    block = np.concatenate([record.samples for record in records], axis=0)
    values = np.ascontiguousarray(block[:, list(output.columns)].T)[None]
    return snapshot_from_array(
        values,
        producer=producer,
        signal=output.name,
        cell_axes=(
            _channel_axis(producer, output.name, output.channel_labels),
            _sample_axis(producer, output.name, sample_interval_seconds, block.shape[0]),
        ),
        value_unit=output.unit,
        generation=str(getattr(generation, "value", generation)),
        revision=int(revision),
    )


def _strict_event_ordinals(
    records: Sequence[WaveformRecord],
    *,
    expected_start: int,
    records_per_event: int,
) -> tuple[WaveformRecord, ...]:
    event = tuple(records)
    expected = tuple(range(expected_start, expected_start + records_per_event))
    observed = tuple(int(record.source_ordinal) for record in event)
    if len(event) != records_per_event or observed != expected:
        raise RuntimeError(
            "waveform event source ordinals are not contiguous: "
            f"expected {expected}, received {observed}"
        )
    return event


def _strict_terminal(
    terminal: WaveformCaptureTerminalRecord,
    *,
    expected_records: int,
    stopped: bool = False,
) -> WaveformCaptureTerminalRecord:
    if not (terminal.source_stopped and terminal.no_more_records and terminal.joined):
        raise RuntimeError(
            "waveform terminal evidence is incomplete: the capture did not stop, "
            "drain and join"
        )
    produced = terminal.produced_count
    if produced < expected_records or (produced != expected_records and not stopped):
        raise RuntimeError(
            "waveform terminal count differs from completed events: "
            f"completed events account for {expected_records} record(s), the "
            f"source produced {produced} (a partial event may be present)"
        )
    return terminal


def _working_point_snapshot(point: WaveformWorkingPoint) -> dict[str, object]:
    return {
        "acquisition_mode": str(point.acquisition_mode),
        "sample_interval_seconds": float(point.sample_interval_seconds),
        "record_samples": int(point.record_samples),
        "outputs": [
            {
                "name": output.name,
                "unit": output.unit,
                "channels": list(output.channel_labels),
            }
            for output in point.outputs
        ],
        "settings": dict(point.settings),
    }


@dataclass(frozen=True)
class WaveformMeasurementRequest:
    """One frozen selection: which source, how many events, how big each is."""

    sampler_key: str
    repeat: int
    records_per_event: int

    def __post_init__(self) -> None:
        key = str(self.sampler_key).strip()
        if not key:
            raise ValueError("sampler_key must be non-empty")
        if int(self.repeat) < 0:
            raise ValueError("repeat must be non-negative")
        if int(self.records_per_event) <= 0:
            raise ValueError("records_per_event must be positive")
        object.__setattr__(self, "sampler_key", key)
        object.__setattr__(self, "repeat", int(self.repeat))
        object.__setattr__(self, "records_per_event", int(self.records_per_event))


class FiniteCapture:
    """An armed finite capture: ``repeat`` events, each read as one exact group."""

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
        self.completed_events = 0
        self.stopped = False
        self.terminal: WaveformCaptureTerminalRecord | None = None

    def collect(
        self,
        *,
        commit_event: Callable[[tuple[WaveformRecord, ...], int], None] | None = None,
    ) -> int:
        """Read every event and publish it; answers how many events were kept."""

        if self.closed:
            raise RuntimeError("finite capture is closed")
        if commit_event is None:
            if not self.owns_generation:
                raise TypeError("hosted finite capture requires commit_event")
            commit_event = self.node._commit_direct_event
        try:
            for index in range(self.node.repeat):
                event = self.next_event()
                if event is None:
                    break
                commit_event(event, index)
            self.close()
        except BaseException:
            if not self.closed:
                self.node.sampler.finish_record_capture()
                self.closed = True
            if self.owns_generation:
                self.node.signal_plane.retire(self.node)
            raise
        if self.owns_generation:
            if self.completed_events:
                self.node.signal_plane.seal_committed(
                    self.node, cut_short=self.completed_events < self.node.repeat
                )
            else:
                self.node.signal_plane.retire(self.node)
        return self.completed_events

    def next_event(self) -> tuple[WaveformRecord, ...] | None:
        """The next complete event, or None if asked to stop."""

        if self.closed:
            raise RuntimeError("finite capture is closed")
        wanted = self.node.records_per_event
        records: tuple[WaveformRecord, ...] = ()
        timeout = float(self.node.sampler.timeout)
        deadline = monotonic() + timeout
        while len(records) < wanted:
            if self.should_stop is not None and self.should_stop():
                self.stopped = True
                return None
            arrived = self.node.sampler.read_records(
                wanted - len(records),
                timeout=min(_CANCEL_RESPONSE_SECONDS, max(0.0, deadline - monotonic())),
                exact=False,
            )
            if arrived:
                records += tuple(arrived)
                deadline = monotonic() + timeout
                continue
            if monotonic() >= deadline:
                break
        if len(records) != wanted:
            raise RuntimeError(
                f"the waveform source returned {len(records)} record(s) of a "
                f"{wanted}-record event before its {timeout:g} s timeout"
            )
        event = _strict_event_ordinals(
            records,
            expected_start=self.completed_events * wanted,
            records_per_event=wanted,
        )
        missing = self.node._missing_samples(event)
        if missing:
            raise RuntimeError(
                f"the waveform source lost {missing} sample(s) inside one event; "
                "its time axis would not be true"
            )
        self.completed_events += 1
        return event

    def close(self) -> WaveformCaptureTerminalRecord:
        if self.terminal is not None:
            return self.terminal
        if self.closed:
            raise RuntimeError("finite capture closed without terminal evidence")
        terminal = _strict_terminal(
            self.node.sampler.finish_record_capture(),
            expected_records=self.completed_events * self.node.records_per_event,
            stopped=self.stopped,
        )
        self.closed = True
        self.terminal = terminal
        return terminal


class MonitorCapture:
    """A repeat-zero monitor: every complete event replaces the last."""

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
        self._pending: list[WaveformRecord] = []
        self._revision = 0
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
        """Take what has arrived and publish every complete event in it."""

        if self.closed:
            raise RuntimeError("monitor capture is closed")
        wanted = self.node.records_per_event
        records = self.node.sampler.read_records(
            wanted, timeout=_CANCEL_RESPONSE_SECONDS, exact=False
        )
        published = 0
        for record in records:
            published += self._accept(record)
        return published

    def _accept(self, record: WaveformRecord) -> int:
        """Publish only an aligned, contiguous event; drop what does not line up."""

        wanted = self.node.records_per_event
        ordinal = int(record.source_ordinal)
        pending = self._pending
        if not pending:
            if ordinal % wanted:
                return 0
            pending.append(record)
        else:
            if ordinal != pending[0].source_ordinal + len(pending):
                pending.clear()
                if ordinal % wanted:
                    return 0
            pending.append(record)
        if len(pending) < wanted:
            return 0
        event = _strict_event_ordinals(
            pending, expected_start=int(pending[0].source_ordinal), records_per_event=wanted
        )
        pending.clear()
        if self.node._missing_samples(event):
            return 0
        self._revision += 1
        self._commit_live(self.node._event_outputs(event, revision=self._revision))
        return 1

    def close(self) -> WaveformCaptureTerminalRecord:
        if self.terminal is not None:
            return self.terminal
        terminal = self.node.sampler.finish_record_capture()
        self.closed = True
        self.terminal = terminal
        if self.owns_generation:
            if self._revision:
                self.node.signal_plane.seal_committed(self.node)
            else:
                self.node.signal_plane.retire(self.node)
        return terminal


class WaveformMeasurementNode:
    """Commit each event of a waveform source to its declared signals."""

    def __init__(
        self,
        *,
        sampler: WaveformSource,
        request: WaveformMeasurementRequest,
        signal_plane: object,
        outputs: tuple[DatasetOutputDeclaration, ...],
        producer: str,
    ) -> None:
        if not isinstance(sampler, WaveformSource):
            raise TypeError("sampler must implement WaveformSource")
        if not isinstance(request, WaveformMeasurementRequest):
            raise TypeError("request must be WaveformMeasurementRequest")
        declared = tuple(outputs)
        if not declared or any(
            not isinstance(output, DatasetOutputDeclaration) for output in declared
        ):
            raise TypeError("outputs must be DatasetOutputDeclaration values")
        if signal_plane is None:
            raise TypeError("signal_plane must be supplied by the runtime owner")
        self.sampler = sampler
        self._request = request
        self.signal_plane = signal_plane
        self._outputs = declared
        self.instance_id = str(producer).strip()
        if not self.instance_id:
            raise ValueError("producer must be non-empty")
        self._generation: object | None = None
        self._working_point: WaveformWorkingPoint | None = None
        self._run_record: dict[str, object] | None = None

    @property
    def request(self) -> WaveformMeasurementRequest:
        return self._request

    @property
    def repeat(self) -> int:
        return self._request.repeat

    @property
    def records_per_event(self) -> int:
        return self._request.records_per_event

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
        """Freeze what the source samples as what this run publishes."""

        point = self.sampler.working_point()
        if not isinstance(point, WaveformWorkingPoint):
            raise TypeError("waveform source working_point must return WaveformWorkingPoint")
        published = {output.name for output in point.outputs}
        declared = {output.name for output in self._outputs}
        if published != declared:
            raise RuntimeError(
                f"the source {self.sampler_key!r} publishes {sorted(published)}, "
                f"this measurement declares {sorted(declared)}: pick the node that "
                "matches the device"
            )
        self._working_point = point
        self._run_record = {
            "node": self.instance_id,
            "parameters": {
                "repeat": self.repeat,
                "records_per_event": self.records_per_event,
            },
            "named_devices": {"sampler": self.sampler_key},
            "device_snapshots": {"sampler": _working_point_snapshot(point)},
        }
        return point

    def _missing_samples(self, event: Sequence[WaveformRecord]) -> int:
        """Samples the source lost inside this event, by its own clock; 0 without one.

        An event's time axis says its samples are one interval apart.  A
        source that stamps its records lets that be checked: a stream that
        dropped a packet inside the event would publish a time axis that
        lies by one interval from the gap on, so such an event is not
        published at all.
        """

        first, last = event[0], event[-1]
        if first.device_timestamp_seconds is None or last.device_timestamp_seconds is None:
            return 0
        point = self.working_point
        expected = (len(event) - 1) * point.record_samples * point.sample_interval_seconds
        span = last.device_timestamp_seconds - first.device_timestamp_seconds
        return max(0, int(round((span - expected) / point.sample_interval_seconds)))

    def _event_outputs(
        self,
        event: Sequence[WaveformRecord],
        *,
        revision: int,
        index: int | None = None,
    ) -> dict[str, LiveDatasetOutput]:
        point = self.working_point
        outputs: dict[str, LiveDatasetOutput] = {}
        for declaration in self._outputs:
            snapshot = event_snapshot(
                event,
                output=point.output(declaration.name),
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

    def _commit_direct_event(self, event: tuple[WaveformRecord, ...], index: int) -> None:
        self._commit_direct_outputs(
            self._event_outputs(event, revision=index + 1, index=index)
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
            total = self.repeat * self.records_per_event
            self.sampler.arm(
                total, buffer_record_count=total, timeout=float(self.sampler.timeout)
            )
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
            # Four events of slack: a monitor that falls behind loses the
            # oldest records, and an event that lost one is dropped whole
            # by the ordinal check rather than published with a seam.
            self.sampler.arm(
                None,
                buffer_record_count=4 * self.records_per_event,
                timeout=float(self.sampler.timeout),
            )
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

        def commit_event(event: tuple[WaveformRecord, ...], index: int) -> None:
            context.commit_live(self._event_outputs(event, revision=index + 1, index=index))
            context.report_progress("Capturing", current=index + 1, total=int(self.repeat))

        completed = capture.collect(commit_event=commit_event)
        return {"events": completed, "signals": signals}


__all__ = [
    "FiniteCapture",
    "MonitorCapture",
    "WaveformMeasurementNode",
    "WaveformMeasurementRequest",
    "event_snapshot",
    "waveform_authoring_schema",
]
