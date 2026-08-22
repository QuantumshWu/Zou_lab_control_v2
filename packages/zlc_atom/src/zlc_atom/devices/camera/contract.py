"""Narrow camera SPI shared by hardware and virtual adapters."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence, runtime_checkable
import time

import numpy as np


def _pair(value: object, field: str, *, positive: bool) -> tuple[int, int]:
    try:
        result = tuple(int(item) for item in value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be a two-integer tuple") from exc
    if len(result) != 2 or any(item <= 0 if positive else item < 0 for item in result):
        raise ValueError(f"{field} must contain two {'positive' if positive else 'non-negative'} integers")
    return result


class CameraAcquisitionMode(str, Enum):
    EXTERNAL_TRIGGERED = "EXTERNAL_TRIGGERED"
    FREE_RUNNING = "FREE_RUNNING"


@dataclass(frozen=True)
class CameraWorkingPoint:
    acquisition_mode: str
    frame_shape_yx: tuple[int, int]
    sensor_shape_yx: tuple[int, int]
    roi_origin_yx: tuple[int, int]
    roi_shape_yx: tuple[int, int]
    binning_yx: tuple[int, int]
    dtype: np.dtype
    count_unit: str
    exposure_seconds: float
    required_external_trigger_interval_seconds: float | None
    external_trigger_integration_start_offset_seconds: float | None
    gain: float
    readout_mode: str
    #: What one count is worth in photoelectrons, and where zero of them
    #: sits: photoelectrons = (count - offset_counts) * electrons_per_count.
    #: The camera's own two numbers, as its configuration states them -- a
    #: virtual sensor knows them exactly, a real one is told what its
    #: datasheet says -- so ``None`` means this sensor does not say, and
    #: nothing downstream may invent it.  Both or neither.
    offset_counts: float | None = None
    electrons_per_count: float | None = None

    def __post_init__(self) -> None:
        for name, positive in (
            ("frame_shape_yx", True),
            ("sensor_shape_yx", True),
            ("roi_shape_yx", True),
            ("binning_yx", True),
            ("roi_origin_yx", False),
        ):
            object.__setattr__(self, name, _pair(getattr(self, name), name, positive=positive))
        if any(origin + extent > sensor for origin, extent, sensor in zip(self.roi_origin_yx, self.roi_shape_yx, self.sensor_shape_yx)):
            raise ValueError("ROI lies outside sensor geometry")
        expected_shape = tuple(extent // factor for extent, factor in zip(self.roi_shape_yx, self.binning_yx))
        if expected_shape != self.frame_shape_yx:
            raise ValueError("frame shape differs from ROI/binning geometry")
        dtype = np.dtype(self.dtype)
        if dtype.kind not in "iuf" or dtype.hasobject or dtype.fields is not None:
            raise TypeError("camera dtype must be a real numeric dtype")
        object.__setattr__(self, "dtype", dtype.newbyteorder("<"))
        if float(self.exposure_seconds) <= 0 or not np.isfinite(self.exposure_seconds):
            raise ValueError("exposure_seconds must be finite and positive")
        if self.required_external_trigger_interval_seconds is not None and float(self.required_external_trigger_interval_seconds) <= 0:
            raise ValueError("external trigger interval must be positive")
        if self.external_trigger_integration_start_offset_seconds is not None and float(self.external_trigger_integration_start_offset_seconds) < 0:
            raise ValueError("integration start offset cannot be negative")
        if float(self.gain) < 0:
            raise ValueError("camera gain cannot be negative")
        if (self.offset_counts is None) != (self.electrons_per_count is None):
            raise ValueError(
                "a photoelectron conversion needs both its offset and its scale"
            )
        if self.electrons_per_count is not None:
            offset = float(self.offset_counts)
            scale = float(self.electrons_per_count)
            if not np.isfinite(offset) or not np.isfinite(scale) or scale <= 0:
                raise ValueError(
                    "electrons_per_count must be finite and positive, and the "
                    "offset finite"
                )
            object.__setattr__(self, "offset_counts", offset)
            object.__setattr__(self, "electrons_per_count", scale)


@dataclass(frozen=True, eq=False)
class CameraFrameRecord:
    """Adapter-owned copy of one frame and its physical metadata."""

    image: np.ndarray
    source_ordinal: int
    produced_count: int | None = None
    frame_stamp: int | None = None
    camera_stamp: int | None = None
    timestamp_seconds: int | None = None
    timestamp_microseconds: int | None = None
    host_received_at_ns: int = 0
    driver_buffer_index: int | None = None
    __hash__ = None

    def __post_init__(self) -> None:
        ordinal = int(self.source_ordinal)
        if ordinal < 0:
            raise ValueError("source_ordinal must be non-negative")
        object.__setattr__(self, "source_ordinal", ordinal)
        for name in ("produced_count", "timestamp_seconds", "timestamp_microseconds", "driver_buffer_index"):
            value = getattr(self, name)
            if value is not None and int(value) < 0:
                raise ValueError(f"{name} must be non-negative")
            if value is not None:
                object.__setattr__(self, name, int(value))
        if self.timestamp_microseconds is not None and self.timestamp_microseconds >= 1_000_000:
            raise ValueError("timestamp_microseconds must be less than 1_000_000")
        if (self.timestamp_seconds is None) != (self.timestamp_microseconds is None):
            raise ValueError("timestamp seconds and microseconds must appear together")
        host = int(self.host_received_at_ns or time.time_ns())
        if host <= 0:
            raise ValueError("host_received_at_ns must be positive")
        object.__setattr__(self, "host_received_at_ns", host)
        # Owned as BYTES, not as a read-only ndarray.  Both spellings cost the
        # same single copy here, but a read-only view over an owning array can
        # be made writable again, so every downstream boundary copied the whole
        # frame a second time to be sure of it.  A view whose base chain ends in
        # bytes cannot acquire a writable buffer, and zlc_data recognises that
        # and keeps the view -- one copy per frame instead of two.
        array = np.asarray(self.image)
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
        owned = np.frombuffer(
            np.ascontiguousarray(array).tobytes(), dtype=array.dtype
        ).reshape(array.shape)
        owned.setflags(write=False)
        object.__setattr__(self, "image", owned)


@dataclass(frozen=True)
class CameraCaptureTerminalRecord:
    produced_count: int
    source_stopped: bool
    no_more_frames: bool
    joined: bool

    def __post_init__(self) -> None:
        if int(self.produced_count) < 0:
            raise ValueError("produced_count must be non-negative")
        object.__setattr__(self, "produced_count", int(self.produced_count))
        if any(type(getattr(self, name)) is not bool for name in ("source_stopped", "no_more_frames", "joined")):
            raise TypeError("terminal proof flags must be bool")


@runtime_checkable
class CameraAdapter(Protocol):
    @property
    def timeout(self) -> float: ...

    #: ``(offset_counts, electrons_per_count)`` or None when this sensor
    #: states no conversion.  Answered from the device's own configuration,
    #: so it can be asked of a camera that is not open -- which is what lets
    #: a form offer the unit before anything has been armed.
    @property
    def photoelectron_conversion(self) -> tuple[float, float] | None: ...

    # A camera has a state: how long it integrates, and which part of the
    # sensor it reads.  The two are owned by different people -- the geometry
    # by the bench, which tunes an ROI around the traps and needs it to
    # survive; the exposure by whatever run is playing, because it has to fit
    # between that run's camera triggers -- and they are set separately for
    # exactly that reason.  One call that demanded both is what let a task
    # replace a tuned ROI with the full sensor because it only meant to state
    # an exposure, and left an exposure so long that the third trigger of a
    # cycle arrived while the sensor was still busy.
    #
    # Nothing here is named after a measurement.  A camera is a shared piece
    # of hardware in a state, not an accessory of whoever last called it.

    def set_exposure_seconds(self, seconds: float) -> CameraWorkingPoint: ...

    def set_roi(
        self, roi_xywh: tuple[int, int, int, int] | None
    ) -> CameraWorkingPoint: ...

    def working_point(self) -> CameraWorkingPoint: ...

    def arm(self, frames: int | None, *, source_group_sizes: tuple[int, ...] | None, buffer_frame_count: int, timeout: float) -> None: ...

    def read_frame_records(self, n: int, *, timeout: float, exact: bool) -> Sequence[CameraFrameRecord]: ...

    def finish_record_capture(self) -> CameraCaptureTerminalRecord: ...

    def capture_state(self) -> bool: ...


__all__ = [
    "CameraAcquisitionMode",
    "CameraAdapter",
    "CameraCaptureTerminalRecord",
    "CameraFrameRecord",
    "CameraWorkingPoint",
]
