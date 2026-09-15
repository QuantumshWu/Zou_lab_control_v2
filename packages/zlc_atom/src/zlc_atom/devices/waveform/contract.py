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

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Protocol, Sequence, runtime_checkable
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


@dataclass(frozen=True)
class WaveformWorkingPoint:
    """How the source samples right now, frozen by a measurement when it arms."""

    acquisition_mode: str
    sample_interval_seconds: float
    record_samples: int
    outputs: tuple[WaveformOutput, ...]
    #: The instrument's own read-back of its settings, as archive-ready
    #: plain values: a scope's time per division, an IMU's packet rate.
    settings: Mapping[str, object]

    def __post_init__(self) -> None:
        mode = str(getattr(self.acquisition_mode, "value", self.acquisition_mode))
        WaveformAcquisitionMode(mode)
        interval = float(self.sample_interval_seconds)
        if not np.isfinite(interval) or interval <= 0.0:
            raise ValueError("sample_interval_seconds must be finite and positive")
        samples = int(self.record_samples)
        if samples <= 0:
            raise ValueError("record_samples must be positive")
        outputs = tuple(self.outputs)
        if not outputs or any(not isinstance(output, WaveformOutput) for output in outputs):
            raise TypeError("working point outputs must be WaveformOutput values")
        names = [output.name for output in outputs]
        if len(set(names)) != len(names):
            raise ValueError("waveform outputs must have distinct names")
        columns = [column for output in outputs for column in output.columns]
        if len(set(columns)) != len(columns):
            raise ValueError("one record column cannot belong to two outputs")
        if not isinstance(self.settings, Mapping):
            raise TypeError("working point settings must be a mapping")
        object.__setattr__(self, "acquisition_mode", mode)
        object.__setattr__(self, "sample_interval_seconds", interval)
        object.__setattr__(self, "record_samples", samples)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "settings", dict(self.settings))

    @property
    def column_count(self) -> int:
        return 1 + max(column for output in self.outputs for column in output.columns)

    def output(self, name: str) -> WaveformOutput:
        for output in self.outputs:
            if output.name == name:
                return output
        raise KeyError(f"waveform source publishes no output {name!r}")


@dataclass(frozen=True, eq=False)
class WaveformRecord:
    """Source-owned copy of one record: ``(record_samples, columns)`` float32."""

    samples: np.ndarray
    source_ordinal: int
    host_received_at_ns: int = 0
    __hash__ = None

    def __post_init__(self) -> None:
        ordinal = int(self.source_ordinal)
        if ordinal < 0:
            raise ValueError("source_ordinal must be non-negative")
        object.__setattr__(self, "source_ordinal", ordinal)
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


@runtime_checkable
class WaveformSource(Protocol):
    @property
    def timeout(self) -> float: ...

    def working_point(self) -> WaveformWorkingPoint: ...

    def arm(
        self, records: int | None, *, buffer_record_count: int, timeout: float
    ) -> None: ...

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
    "WaveformSource",
    "WaveformWorkingPoint",
]
