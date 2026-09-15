"""Virtual waveform source: a paced sampler whose samples are injected."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Callable

import numpy as np

from zlc_atom.devices.waveform.contract import (
    WaveformAcquisitionMode,
    WaveformCaptureTerminalRecord,
    WaveformOutput,
    WaveformRecord,
    WaveformWorkingPoint,
)


@dataclass(frozen=True)
class VirtualWaveformConfig:
    """How fast it samples, how many samples a record holds, and what the columns are."""

    sample_rate_hz: float
    record_samples: int
    outputs: tuple[WaveformOutput, ...]

    def __post_init__(self) -> None:
        rate = float(self.sample_rate_hz)
        if not np.isfinite(rate) or rate <= 0.0:
            raise ValueError("sample_rate_hz must be finite and positive")
        if int(self.record_samples) <= 0:
            raise ValueError("record_samples must be positive")
        outputs = tuple(self.outputs)
        if not outputs:
            raise ValueError("a virtual waveform source publishes at least one output")
        object.__setattr__(self, "sample_rate_hz", rate)
        object.__setattr__(self, "record_samples", int(self.record_samples))
        object.__setattr__(self, "outputs", outputs)


class VirtualWaveformSource:
    """A free-running sampler that keeps real time; the physics is injected.

    ``sample_source`` is handed the sample times of one record (seconds
    from arm) and answers the ``(record_samples, columns)`` array those
    samples read.  Arming, the bounded buffer, ordinals, the record clock
    and the terminal are the same code a hardware source runs, which is
    what makes a virtual bench able to catch a measurement that mishandles
    them.
    """

    def __init__(
        self,
        config: VirtualWaveformConfig,
        *,
        sample_source: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        if sample_source is None or not callable(sample_source):
            raise TypeError("virtual waveform source requires an injected sample_source")
        self.config = config
        self._sample_source = sample_source
        self._columns = 1 + max(
            column for output in config.outputs for column in output.columns
        )
        self._condition = threading.Condition()
        self._queue: deque[WaveformRecord] = deque()
        self._armed = False
        self._accepting = False
        self._expected: int | None = None
        self._buffer_record_count = 1
        self._next_ordinal = 0
        self._produced_count = 0
        self._worker: threading.Thread | None = None
        self._worker_stop: threading.Event | None = None
        self._worker_error: BaseException | None = None
        self._terminal: WaveformCaptureTerminalRecord | None = None

    @property
    def timeout(self) -> float:
        return 2.0

    def working_point(self) -> WaveformWorkingPoint:
        return WaveformWorkingPoint(
            WaveformAcquisitionMode.FREE_RUNNING,
            1.0 / self.config.sample_rate_hz,
            self.config.record_samples,
            self.config.outputs,
            {
                "sample_rate_hz": self.config.sample_rate_hz,
                "record_samples": self.config.record_samples,
            },
        )

    def arm(
        self, records: int | None, *, buffer_record_count: int, timeout: float
    ) -> None:
        del timeout
        buffer_count = int(buffer_record_count)
        if buffer_count <= 0:
            raise ValueError("buffer_record_count must be positive")
        expected = None if records is None else int(records)
        if expected is not None and (expected <= 0 or buffer_count != expected):
            raise ValueError("a finite arm buffers exactly the records it expects")
        with self._condition:
            if self._armed:
                raise RuntimeError("virtual waveform source is already armed")
            if self._worker is not None and self._worker.is_alive():
                raise RuntimeError("previous virtual waveform worker is still running")
            self._queue.clear()
            self._armed = True
            self._accepting = True
            self._expected = expected
            self._buffer_record_count = buffer_count
            self._next_ordinal = 0
            self._produced_count = 0
            self._worker_error = None
            self._terminal = None
            stop = threading.Event()
            worker = threading.Thread(
                target=self._produce,
                args=(stop,),
                name="zlc-virtual-waveform-producer",
                daemon=True,
            )
            self._worker_stop = stop
            self._worker = worker
            try:
                worker.start()
            except BaseException:
                self._worker = None
                self._worker_stop = None
                self._armed = False
                self._accepting = False
                raise

    def _produce(self, stop: threading.Event) -> None:
        interval = 1.0 / self.config.sample_rate_hz
        samples = self.config.record_samples
        offsets = np.arange(samples, dtype=np.float64) * interval
        record_seconds = samples * interval
        started = time.monotonic()
        due = started
        try:
            while not stop.is_set():
                with self._condition:
                    if not self._accepting:
                        break
                    while self._accepting and not stop.is_set():
                        remaining = due - time.monotonic()
                        if remaining <= 0.0:
                            break
                        self._condition.wait(timeout=remaining)
                    if not self._accepting or stop.is_set():
                        break
                    ordinal = self._next_ordinal
                    self._next_ordinal += 1
                first = due - started
                values = np.asarray(self._sample_source(first + offsets), dtype=np.float32)
                if values.shape != (samples, self._columns):
                    raise ValueError(
                        "virtual sample source returned the wrong shape: "
                        f"{values.shape} for {(samples, self._columns)}"
                    )
                record = WaveformRecord(values, ordinal, first, time.time_ns())
                with self._condition:
                    if stop.is_set() or not self._armed:
                        break
                    self._produced_count += 1
                    while len(self._queue) >= self._buffer_record_count:
                        self._queue.popleft()
                    self._queue.append(record)
                    if self._expected is not None and self._produced_count >= self._expected:
                        self._accepting = False
                    self._condition.notify_all()
                due += record_seconds
        except BaseException as error:  # noqa: BLE001 -- surfaced to the reader of records
            with self._condition:
                self._worker_error = error
                self._accepting = False
                self._condition.notify_all()
        finally:
            with self._condition:
                if self._worker is threading.current_thread():
                    self._worker = None
                    self._worker_stop = None
                self._condition.notify_all()

    def read_records(
        self, n: int, *, timeout: float, exact: bool
    ) -> list[WaveformRecord]:
        requested = int(n)
        if requested <= 0:
            raise ValueError("n must be positive")
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while len(self._queue) < requested:
                if self._worker_error is not None:
                    raise RuntimeError("virtual waveform worker failed") from self._worker_error
                running = self._worker is not None and self._worker.is_alive()
                if not self._armed or (not running and not self._accepting):
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._condition.wait(remaining)
            if exact and len(self._queue) < requested:
                raise TimeoutError("virtual waveform source did not produce the requested exact records")
            count = requested if exact else min(requested, len(self._queue))
            return [self._queue.popleft() for _ in range(count)]

    def finish_record_capture(self) -> WaveformCaptureTerminalRecord:
        with self._condition:
            if self._terminal is not None:
                return self._terminal
            if not self._armed:
                self._terminal = WaveformCaptureTerminalRecord(0, True, True, True)
                return self._terminal
            self._accepting = False
            worker = self._worker
            stop = self._worker_stop
            if stop is not None:
                stop.set()
            self._condition.notify_all()
        if worker is not None:
            worker.join(timeout=2.0)
        with self._condition:
            if worker is not None and worker.is_alive():
                raise RuntimeError("virtual waveform producer did not join")
            self._armed = False
            self._terminal = WaveformCaptureTerminalRecord(
                self._produced_count, True, not self._queue, True
            )
            self._condition.notify_all()
            if self._worker_error is not None:
                raise RuntimeError("virtual waveform worker failed") from self._worker_error
            return self._terminal

    def capture_state(self) -> bool:
        with self._condition:
            return self._armed

    def close(self) -> None:
        if self.capture_state():
            self.finish_record_capture()
        with self._condition:
            self._accepting = False
            self._queue.clear()
            self._condition.notify_all()

    @property
    def produced_count(self) -> int:
        with self._condition:
            return self._produced_count


__all__ = ["VirtualWaveformConfig", "VirtualWaveformSource"]
