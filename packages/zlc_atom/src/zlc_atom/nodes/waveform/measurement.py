"""Finite exact and live-monitor capture of a waveform source: one record, one shot.

This package is a LIBRARY, not a node: it carries no ``logic_node.py``.
Two nodes stand on it -- the IMU measurement and the scope measurement --
and what differs between them is only what they publish: an IMU packet is
four quantities, a scope acquisition is one.  The capture, the read
cadence and the commit are the same for both, and live here once.

Every record the measurement takes is one shot, published the moment it
is read: an IMU packet becomes ``(1) x () x (channel)`` per quantity, a
scope acquisition ``(1) x () x (channel, time)`` with the samples on a
READOUT_EVENT axis whose coordinates are seconds from the trigger.  The
history of shots is the Runtime's: every output declares
``index_by_source``, so a Rolling panel leases a window and the plane
keeps the last N shots by their own sequence.

What the measurement is told, besides how many shots, is HOW OFTEN TO READ
THE HARDWARE.  Zero means every record the source produces is a shot; an
interval means that at each due time the newest record is the reading and
whatever arrived in between is not published -- a magnetometer streaming
at 400 Hz read every 100 ms is ten shots a second, each the field right
then.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from math import ceil
from time import monotonic, sleep

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
)


#: How often a capture comes back to see whether it has been asked to stop;
#: the read length is the cancel latency, and it belongs to the loop.
_CANCEL_RESPONSE_SECONDS = 0.05


def waveform_authoring_schema(*, read_interval_seconds: float) -> AuthoringSchema:
    """What a waveform measurement is told: how many shots, how often to read."""

    return AuthoringSchema(
        (
            AuthoringField("repeat", "int", "Repeat", 0, minimum=0),
            AuthoringField(
                "read_interval_seconds",
                "float",
                "Read interval",
                float(read_interval_seconds),
                minimum=0.0,
                unit="s",
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


def _strict_terminal(
    terminal: WaveformCaptureTerminalRecord,
    *,
    expected_records: int | None,
    stopped: bool = False,
) -> WaveformCaptureTerminalRecord:
    """The device's terminal, checked against what the capture took.

    ``expected_records`` is None for a capture that read at its own
    cadence: it took the newest record at each due time and let the rest
    go, so the source's count says nothing about its shots.
    """

    if not (terminal.source_stopped and terminal.joined):
        raise RuntimeError(
            "waveform terminal evidence is incomplete: the capture did not stop and join"
        )
    if expected_records is None:
        # What the source still holds unread is what a sampling read let go.
        return terminal
    if not terminal.no_more_records:
        raise RuntimeError(
            "waveform terminal evidence is incomplete: the source still holds "
            "records a contiguous capture never took"
        )
    produced = terminal.produced_count
    if produced < expected_records or (produced != expected_records and not stopped):
        raise RuntimeError(
            "waveform terminal count differs from completed shots: "
            f"{expected_records} shot(s) were published, the source produced "
            f"{produced} record(s)"
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
        if not np.isfinite(interval) or interval < 0.0:
            raise ValueError("read_interval_seconds must be finite and non-negative")
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
            if not self.closed:
                self.node.sampler.finish_record_capture()
                self.closed = True
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

        At zero interval it is the next record, and the records must be
        contiguous from the arm: a record the source lost is a shot this
        run cannot account for.  At an interval it is the newest record at
        the due time; what arrived before it is let go.
        """

        if self.closed:
            raise RuntimeError("finite capture is closed")
        node = self.node
        interval = node.read_interval_seconds
        timeout = float(node.sampler.timeout)
        deadline = monotonic() + timeout
        while True:
            if self.should_stop is not None and self.should_stop():
                self.stopped = True
                return None
            if interval > 0.0:
                record = _newest_at_due(node, self._next_due)
            else:
                arrived = node.sampler.read_records(
                    1,
                    timeout=min(_CANCEL_RESPONSE_SECONDS, max(0.0, deadline - monotonic())),
                    exact=False,
                )
                record = arrived[0] if arrived else None
            if record is not None:
                break
            if monotonic() >= deadline:
                raise RuntimeError(
                    f"the waveform source delivered no record within its {timeout:g} s timeout"
                )
        if interval > 0.0:
            self._next_due = _due_after(self._next_due, interval)
        elif int(record.source_ordinal) != self.completed_shots:
            raise RuntimeError(
                "waveform records are not contiguous: expected ordinal "
                f"{self.completed_shots}, received {int(record.source_ordinal)}"
            )
        self.completed_shots += 1
        return record

    def close(self) -> WaveformCaptureTerminalRecord:
        if self.terminal is not None:
            return self.terminal
        if self.closed:
            raise RuntimeError("finite capture closed without terminal evidence")
        terminal = _strict_terminal(
            self.node.sampler.finish_record_capture(),
            expected_records=(
                None if self.node.read_interval_seconds > 0.0 else self.completed_shots
            ),
            stopped=self.stopped,
        )
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
        interval = node.read_interval_seconds
        if interval <= 0.0:
            records = node.sampler.read_records(
                node.read_batch, timeout=_CANCEL_RESPONSE_SECONDS, exact=False
            )
            for record in records:
                self._publish(record)
            return len(records)
        newest = _newest_at_due(node, self._next_due)
        if newest is None:
            return 0
        self._publish(newest)
        self._next_due = _due_after(self._next_due, interval)
        return 1

    def _publish(self, record: WaveformRecord) -> None:
        self._revision += 1
        self._commit_live(self.node._shot_outputs(record, revision=self._revision))

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
    """Commit each shot of a waveform source to its declared signals."""

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

        point = self.working_point
        per_slice = _CANCEL_RESPONSE_SECONDS / (
            point.record_samples * point.sample_interval_seconds
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
                "read_interval_seconds": self.read_interval_seconds,
            },
            "named_devices": {"sampler": self.sampler_key},
            "device_snapshots": {"sampler": _working_point_snapshot(point)},
        }
        return point

    def _shot_outputs(
        self,
        record: WaveformRecord,
        *,
        revision: int,
        index: int | None = None,
    ) -> dict[str, LiveDatasetOutput]:
        point = self.working_point
        outputs: dict[str, LiveDatasetOutput] = {}
        for declaration in self._outputs:
            snapshot = shot_snapshot(
                record,
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

    def _commit_direct_shot(self, record: WaveformRecord, index: int) -> None:
        self._commit_direct_outputs(
            self._shot_outputs(record, revision=index + 1, index=index)
        )

    def _arm(self, records: int | None) -> None:
        """Arm for exactly ``records`` contiguous records, or for sampling.

        Sampling reads the newest record at each due time and drains the
        rest, so the buffer only has to hold a few slices' worth; a source
        that outruns it drops the oldest, which a sampling read would have
        let go anyway.
        """

        if records is not None:
            self.sampler.arm(records, buffer_record_count=records)
            return
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
            self._arm(None if self.read_interval_seconds > 0.0 else self.repeat)
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
            self._arm(None)
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
    "FiniteCapture",
    "MonitorCapture",
    "WaveformMeasurementNode",
    "WaveformMeasurementRequest",
    "shot_snapshot",
    "waveform_authoring_schema",
]
