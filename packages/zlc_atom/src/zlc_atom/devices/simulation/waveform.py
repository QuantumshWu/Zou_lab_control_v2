"""Virtual waveform source: a paced sampler whose samples are injected."""

from __future__ import annotations

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
    WaveformRecordQueue,
    WaveformWorkingPoint,
)

# The producer sleeps to its record clock in slices no longer than this, so a
# stop is noticed within one slice whatever the record period is.  It sleeps
# rather than waiting on a lock because a timed lock wait has the OS timer
# tick as its resolution (15 ms on Windows) and would bunch the records of a
# fast sampler into bursts; the sleep keeps under a millisecond.
_STOP_RESPONSE_SECONDS = 0.05


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
    samples read.  The arming, the bounded ring, the ordinals and the
    terminal are the same ring a hardware source runs, which is what makes
    a virtual bench able to catch a measurement that mishandles them.
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
        self._records = WaveformRecordQueue(
            "the virtual waveform source", join_timeout_seconds=self.timeout
        )

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

    def arm(self, records: int | None, *, buffer_record_count: int) -> None:
        self._records.arm(
            records,
            buffer_record_count=buffer_record_count,
            worker=lambda stop: threading.Thread(
                target=self._produce,
                args=(stop,),
                name="zlc-virtual-waveform-producer",
                daemon=True,
            ),
        )

    def _produce(self, stop: threading.Event) -> None:
        interval = 1.0 / self.config.sample_rate_hz
        samples = self.config.record_samples
        offsets = np.arange(samples, dtype=np.float64) * interval
        record_seconds = samples * interval
        started = time.monotonic()
        due = started
        try:
            while not stop.is_set() and self._records.accepting:
                remaining = due - time.monotonic()
                if remaining > 0.0:
                    time.sleep(min(remaining, _STOP_RESPONSE_SECONDS))
                    continue
                values = np.asarray(
                    self._sample_source(due - started + offsets), dtype=np.float32
                )
                if values.shape != (samples, self._columns):
                    raise ValueError(
                        "virtual sample source returned the wrong shape: "
                        f"{values.shape} for {(samples, self._columns)}"
                    )
                self._records.push(values, time.time_ns())
                due += record_seconds
        except BaseException as error:  # noqa: BLE001 -- surfaced to the reader of records
            self._records.fail(error)

    def read_records(
        self, n: int, *, timeout: float, exact: bool
    ) -> list[WaveformRecord]:
        return self._records.read(n, timeout=timeout, exact=exact)

    def finish_record_capture(self) -> WaveformCaptureTerminalRecord:
        return self._records.finish()

    def capture_state(self) -> bool:
        return self._records.armed

    def close(self) -> None:
        if self._records.armed:
            self._records.finish()

    @property
    def produced_count(self) -> int:
        return self._records.produced_count


__all__ = ["VirtualWaveformConfig", "VirtualWaveformSource"]
