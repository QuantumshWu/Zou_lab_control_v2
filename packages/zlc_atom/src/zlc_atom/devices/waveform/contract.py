"""Narrow sampled-waveform SPI shared by hardware and virtual sources.

A waveform source is a camera for time: it samples a few channels at a fixed
interval and hands back RECORDS -- one packet of an IMU stream, one triggered
acquisition of an oscilloscope -- numbered from zero for the life of one arm.
What the columns of a record mean, in which unit, and how they group into
the signals a measurement publishes is the source's own statement, made once
per working point, because a nine-axis IMU carries four quantities in one
packet and a scope carries one quantity on four channels.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import threading
from typing import Callable, Mapping, Protocol, Sequence, runtime_checkable
import time

import numpy as np
from zlc_data.units import resolve_unit


class WaveformAcquisitionMode(str, Enum):
    EXTERNAL_TRIGGERED = "EXTERNAL_TRIGGERED"
    FREE_RUNNING = "FREE_RUNNING"


@dataclass(frozen=True)
class WaveformOutput:
    """One published quantity: which record columns, what they are called, in what unit."""

    name: str
    unit: str
    channel_labels: tuple[str, ...]
    columns: tuple[int, ...]

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("waveform output name must be non-empty")
        unit = str(self.unit).strip()
        resolve_unit(unit)
        labels = tuple(str(label).strip() for label in self.channel_labels)
        columns = tuple(int(column) for column in self.columns)
        if not labels or any(not label for label in labels):
            raise ValueError(f"waveform output {name!r} needs non-empty channel labels")
        if len(labels) != len(columns) or len(set(columns)) != len(columns):
            raise ValueError(
                f"waveform output {name!r} needs one distinct record column per channel"
            )
        if any(column < 0 for column in columns):
            raise ValueError("waveform output columns must be non-negative")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "channel_labels", labels)
        object.__setattr__(self, "columns", columns)


def validate_waveform_outputs(outputs: Sequence[WaveformOutput]) -> tuple[WaveformOutput, ...]:
    """The outputs a source publishes, checked once: named apart, columns apart."""

    declared = tuple(outputs)
    if not declared or any(not isinstance(output, WaveformOutput) for output in declared):
        raise TypeError("waveform outputs must be WaveformOutput values")
    names = [output.name for output in declared]
    if len(set(names)) != len(names):
        raise ValueError("waveform outputs must have distinct names")
    columns = [column for output in declared for column in output.columns]
    if len(set(columns)) != len(columns):
        raise ValueError("one record column cannot belong to two outputs")
    return declared


@dataclass(frozen=True)
class WaveformWorkingPoint:
    """How the source samples right now, frozen by a measurement when it arms.

    What the source publishes and how long its record is are facts about
    the instrument, known from open (``WaveformSource.outputs`` and
    ``record_samples``); this is the part that a knob can move between one
    run and the next -- how fast it samples, and the settings that say so.
    """

    acquisition_mode: str
    sample_interval_seconds: float
    #: The instrument's own read-back of its settings, as archive-ready
    #: plain values: a scope's time per division, an IMU's packet rate.
    settings: Mapping[str, object]

    def __post_init__(self) -> None:
        mode = str(getattr(self.acquisition_mode, "value", self.acquisition_mode))
        WaveformAcquisitionMode(mode)
        interval = float(self.sample_interval_seconds)
        if not np.isfinite(interval) or interval <= 0.0:
            raise ValueError("sample_interval_seconds must be finite and positive")
        if not isinstance(self.settings, Mapping):
            raise TypeError("working point settings must be a mapping")
        object.__setattr__(self, "acquisition_mode", mode)
        object.__setattr__(self, "sample_interval_seconds", interval)
        object.__setattr__(self, "settings", dict(self.settings))


@dataclass(frozen=True, eq=False)
class WaveformRecord:
    """Source-owned copy of one record: ``(record_samples, columns)`` float32.

    ``time_seconds`` is when the record was taken, on the clock the source
    keeps: the module's own packet timestamp for an IMU that has one, the
    host's monotonic clock for a scope read over a link.  Only differences
    between records of one capture mean anything; the measurement anchors
    them at its first shot.
    """

    samples: np.ndarray
    source_ordinal: int
    time_seconds: float
    host_received_at_ns: int = 0
    __hash__ = None

    def __post_init__(self) -> None:
        ordinal = int(self.source_ordinal)
        if ordinal < 0:
            raise ValueError("source_ordinal must be non-negative")
        object.__setattr__(self, "source_ordinal", ordinal)
        seconds = float(self.time_seconds)
        if not np.isfinite(seconds):
            raise ValueError("time_seconds must be finite")
        object.__setattr__(self, "time_seconds", seconds)
        host = int(self.host_received_at_ns or time.time_ns())
        if host <= 0:
            raise ValueError("host_received_at_ns must be positive")
        object.__setattr__(self, "host_received_at_ns", host)
        array = np.asarray(self.samples, dtype=np.dtype("<f4"))
        if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] <= 0:
            raise ValueError("waveform samples must be a (record_samples, columns) array")
        # Owned as BYTES, as a camera frame is: a view whose base chain ends
        # in bytes cannot acquire a writable buffer, so zlc_data keeps the
        # view instead of copying the record again at every boundary.
        owned = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(
            array.shape
        )
        object.__setattr__(self, "samples", owned)


@dataclass(frozen=True)
class WaveformCaptureTerminalRecord:
    produced_count: int
    source_stopped: bool
    no_more_records: bool
    joined: bool

    def __post_init__(self) -> None:
        if int(self.produced_count) < 0:
            raise ValueError("produced_count must be non-negative")
        object.__setattr__(self, "produced_count", int(self.produced_count))
        if any(
            type(getattr(self, name)) is not bool
            for name in ("source_stopped", "no_more_records", "joined")
        ):
            raise TypeError("terminal proof flags must be bool")


class WaveformRecordQueue:
    """The armed ring of records every source hands its records through.

    Arming says how many records are expected and how many the ring holds.
    The source pushes each record as it produces it; the ring numbers it,
    drops the oldest when full, stops accepting once the expected count is
    reached, and answers reads.  A source that acquires on a thread of its
    own per capture hands that thread in at arm and the ring stops and
    joins it at finish; a source whose reader outlives captures (a serial
    stream) hands none, and a failure of that reader is final.
    """

    def __init__(self, what: str, *, join_timeout_seconds: float) -> None:
        self._what = str(what)
        self._join_timeout = float(join_timeout_seconds)
        self._condition = threading.Condition()
        self._queue: deque[WaveformRecord] = deque()
        self._armed = False
        self._accepting = False
        self._expected: int | None = None
        self._buffer_record_count = 1
        self._next_ordinal = 0
        self._produced_count = 0
        self._failure: BaseException | None = None
        self._worker: threading.Thread | None = None
        self._stop: threading.Event | None = None
        self._terminal: WaveformCaptureTerminalRecord | None = None

    @property
    def armed(self) -> bool:
        with self._condition:
            return self._armed

    @property
    def accepting(self) -> bool:
        with self._condition:
            return self._accepting

    @property
    def produced_count(self) -> int:
        with self._condition:
            return self._produced_count

    @property
    def failure(self) -> BaseException | None:
        with self._condition:
            return self._failure

    def arm(
        self,
        records: int | None,
        *,
        buffer_record_count: int,
        worker: Callable[[threading.Event], threading.Thread] | None = None,
    ) -> None:
        """Start a capture; ``worker`` builds the producer thread from its stop event."""

        buffer_count = int(buffer_record_count)
        if buffer_count <= 0:
            raise ValueError("buffer_record_count must be positive")
        expected = None if records is None else int(records)
        if expected is not None and (expected <= 0 or buffer_count != expected):
            raise ValueError("a finite arm buffers exactly the records it expects")
        with self._condition:
            if self._armed:
                raise RuntimeError(f"{self._what} is already armed")
            if self._worker is not None and self._worker.is_alive():
                raise RuntimeError(f"{self._what} is still finishing its previous capture")
            if self._failure is not None and worker is None:
                raise RuntimeError(
                    f"{self._what} failed while producing records"
                ) from self._failure
            self._queue.clear()
            self._armed = True
            self._accepting = True
            self._expected = expected
            self._buffer_record_count = buffer_count
            self._next_ordinal = 0
            self._produced_count = 0
            self._terminal = None
            if worker is None:
                return
            self._failure = None
            stop = threading.Event()
            thread = worker(stop)
            self._worker = thread
            self._stop = stop
            try:
                thread.start()
            except BaseException:
                self._worker = None
                self._stop = None
                self._armed = False
                self._accepting = False
                raise

    def push(
        self, samples: np.ndarray, time_seconds: float, host_received_at_ns: int
    ) -> bool:
        """Number and keep one record; False when the ring is not accepting."""

        with self._condition:
            if not self._accepting:
                return False
            record = WaveformRecord(
                samples, self._next_ordinal, time_seconds, host_received_at_ns
            )
            self._next_ordinal += 1
            self._produced_count += 1
            while len(self._queue) >= self._buffer_record_count:
                self._queue.popleft()
            self._queue.append(record)
            if self._expected is not None and self._produced_count >= self._expected:
                self._accepting = False
            self._condition.notify_all()
            return True

    def fail(self, error: BaseException) -> None:
        """The producer has died; readers learn it, the capture stops accepting."""

        with self._condition:
            self._failure = error
            self._accepting = False
            self._condition.notify_all()

    def read(self, n: int, *, timeout: float, exact: bool) -> list[WaveformRecord]:
        requested = int(n)
        if requested <= 0:
            raise ValueError("n must be positive")
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while len(self._queue) < requested:
                if self._failure is not None:
                    raise RuntimeError(
                        f"{self._what} failed while producing records"
                    ) from self._failure
                if not self._armed or not self._accepting:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._condition.wait(remaining)
            if exact and len(self._queue) < requested:
                raise TimeoutError(f"{self._what} did not deliver the requested exact records")
            count = requested if exact else min(requested, len(self._queue))
            return [self._queue.popleft() for _ in range(count)]

    def finish(self) -> WaveformCaptureTerminalRecord:
        """Stop the capture, join its producer, and say what it produced."""

        with self._condition:
            if self._terminal is not None:
                return self._terminal
            if not self._armed:
                self._terminal = WaveformCaptureTerminalRecord(0, True, True, True)
                return self._terminal
            self._accepting = False
            worker = self._worker
            stop = self._stop
            if stop is not None:
                stop.set()
            self._condition.notify_all()
        if worker is not None:
            worker.join(timeout=self._join_timeout)
        with self._condition:
            if worker is not None and worker.is_alive():
                raise RuntimeError(
                    f"{self._what} did not stop producing within {self._join_timeout:g} s"
                )
            self._worker = None
            self._stop = None
            self._armed = False
            self._terminal = WaveformCaptureTerminalRecord(
                self._produced_count, True, not self._queue, True
            )
            self._condition.notify_all()
            if self._failure is not None:
                raise RuntimeError(
                    f"{self._what} failed while producing records"
                ) from self._failure
            return self._terminal


@runtime_checkable
class WaveformSource(Protocol):
    @property
    def timeout(self) -> float: ...

    @property
    def outputs(self) -> tuple[WaveformOutput, ...]:
        """What one record carries: the quantities, their channels, their units."""
        ...

    @property
    def record_samples(self) -> int:
        """How many samples one record holds: one for a packet, a trace for a scope."""
        ...

    def working_point(self) -> WaveformWorkingPoint: ...

    def arm(self, records: int | None, *, buffer_record_count: int) -> None: ...

    def read_records(
        self, n: int, *, timeout: float, exact: bool
    ) -> Sequence[WaveformRecord]: ...

    def finish_record_capture(self) -> WaveformCaptureTerminalRecord: ...

    def capture_state(self) -> bool: ...


__all__ = [
    "WaveformAcquisitionMode",
    "WaveformCaptureTerminalRecord",
    "WaveformOutput",
    "WaveformRecord",
    "WaveformRecordQueue",
    "WaveformSource",
    "WaveformWorkingPoint",
    "validate_waveform_outputs",
]
