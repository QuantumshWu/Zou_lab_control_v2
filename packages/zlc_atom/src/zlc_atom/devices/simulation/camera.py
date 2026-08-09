"""Virtual camera adapter with asynchronous bounded-buffer semantics."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Callable

import numpy as np

from zlc_atom.devices.camera.contract import (
    CameraCaptureTerminalRecord,
    CameraFrameRecord,
    CameraWorkingPoint,
)
from .world import DEFAULT_SIMULATION_IMAGE_SHAPE_YX


@dataclass(frozen=True)
class VirtualCameraConfig:
    """Geometry and exposure only; imaging physics belongs to SimulationWorld."""

    frame_shape_yx: tuple[int, int] = DEFAULT_SIMULATION_IMAGE_SHAPE_YX
    exposure_seconds: float = 0.02


class VirtualCamera:
    """A trigger-driven camera whose frame source is injected by the world."""

    def __init__(
        self,
        config: VirtualCameraConfig | None = None,
        *,
        frame_source: Callable[[int, float], np.ndarray] | None = None,
    ) -> None:
        if frame_source is None or not callable(frame_source):
            raise TypeError("virtual camera requires an injected frame_source")
        self.config = VirtualCameraConfig() if config is None else config
        shape = tuple(int(item) for item in self.config.frame_shape_yx)
        if len(shape) != 2 or any(item <= 0 for item in shape):
            raise ValueError("frame_shape_yx must contain two positive dimensions")
        if float(self.config.exposure_seconds) <= 0:
            raise ValueError("exposure_seconds must be positive")
        self._frame_source = frame_source
        self._condition = threading.Condition()
        self._sensor_shape_yx = shape
        self._exposure_seconds = float(self.config.exposure_seconds)
        self._roi_xywh = (0, 0, shape[1], shape[0])
        self._queue: deque[CameraFrameRecord] = deque()
        self._trigger_queue: deque[tuple[int, np.ndarray | None]] = deque()
        self._armed = False
        self._accepting = False
        self._expected_frames: int | None = None
        self._buffer_frame_count = 1
        self._next_ordinal = 0
        self._triggered_count = 0
        self._produced_count = 0
        self._worker: threading.Thread | None = None
        self._worker_stop: threading.Event | None = None
        self._worker_error: BaseException | None = None
        self._terminal: CameraCaptureTerminalRecord | None = None

    @property
    def timeout(self) -> float:
        return 2.0

    @property
    def frame_dtype(self) -> np.dtype:
        return np.dtype("<u2")

    def capture_working_point(self) -> CameraWorkingPoint:
        with self._condition:
            exposure = self._exposure_seconds
            x, y, width, height = self._roi_xywh
        return CameraWorkingPoint(
            "EXTERNAL_TRIGGERED",
            (height, width),
            self._sensor_shape_yx,
            (y, x),
            (height, width),
            (1, 1),
            self.frame_dtype,
            "count",
            exposure,
            exposure,
            0.0,
            1.0,
            "virtual-external-trigger",
        )

    def configure_measurement(
        self,
        *,
        exposure_seconds: float,
        roi_xywh: tuple[int, int, int, int] | None,
    ) -> CameraWorkingPoint:
        exposure = float(exposure_seconds)
        if not np.isfinite(exposure) or exposure <= 0:
            raise ValueError("exposure_seconds must be positive and finite")
        sensor_height, sensor_width = self._sensor_shape_yx
        if roi_xywh is None:
            roi = (0, 0, sensor_width, sensor_height)
        else:
            try:
                values = tuple(int(value) for value in roi_xywh)
            except (TypeError, ValueError) as exc:
                raise TypeError("roi_xywh must contain four integers or be None") from exc
            if len(values) != 4:
                raise ValueError("roi_xywh must contain four integers or be None")
            x, y, width, height = values
            x = max(0, min(x, sensor_width - 1))
            y = max(0, min(y, sensor_height - 1))
            width = max(1, min(width, sensor_width - x))
            height = max(1, min(height, sensor_height - y))
            roi = (x, y, width, height)
        with self._condition:
            if self._armed:
                raise RuntimeError("virtual camera settings cannot change while armed")
            self._exposure_seconds = exposure
            self._roi_xywh = roi
        return self.capture_working_point()

    def arm(
        self,
        frames: int | None,
        *,
        source_group_sizes: tuple[int, ...] | None,
        buffer_frame_count: int,
        timeout: float,
    ) -> None:
        del timeout
        buffer_count = int(buffer_frame_count)
        if buffer_count <= 0:
            raise ValueError("buffer_frame_count must be positive")
        if frames is None:
            expected = None
            groups = tuple(int(item) for item in (source_group_sizes or ()))
            if groups and (len(groups) != 1 or groups[0] <= 0):
                raise ValueError("continuous external capture requires one positive source group")
        else:
            expected = int(frames)
            groups = tuple(int(item) for item in (source_group_sizes or ()))
            if expected <= 0 or not groups or sum(groups) != expected or any(item <= 0 for item in groups):
                raise ValueError("finite arm groups must exactly cover frames")
            if buffer_count != expected:
                raise ValueError("finite buffer_frame_count must equal frames")
        with self._condition:
            if self._armed:
                raise RuntimeError("virtual camera is already armed")
            if self._worker is not None and self._worker.is_alive():
                raise RuntimeError("previous virtual camera worker is still running")
            self._queue.clear()
            self._trigger_queue.clear()
            self._armed = True
            self._accepting = True
            self._expected_frames = expected
            self._buffer_frame_count = buffer_count
            self._next_ordinal = 0
            self._triggered_count = 0
            self._produced_count = 0
            self._worker_error = None
            self._terminal = None
            stop = threading.Event()
            # Daemon, unlike the SDK-owning lanes the real cameras keep alive:
            # there is no driver handle here that must be released from its own
            # thread before the process exits, and frames produced during a
            # teardown go nowhere.  Orderly shutdown is `disarm`'s job.  A
            # non-daemon producer buys nothing and costs the worst failure there
            # is -- a process that cannot exit after a crash, so the traceback
            # never reaches anyone and the run reads as a hang.
            worker = threading.Thread(
                target=self._produce,
                args=(stop,),
                name="zlc-virtual-camera-producer",
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
        try:
            while True:
                with self._condition:
                    while not self._trigger_queue and self._accepting and not stop.is_set():
                        self._condition.wait()
                    if self._trigger_queue:
                        ordinal, provided = self._trigger_queue.popleft()
                        exposure = self._exposure_seconds
                        roi = self._roi_xywh
                    elif not self._accepting or stop.is_set():
                        break
                    else:
                        continue
                source = (
                    provided
                    if provided is not None
                    else self._frame_source(ordinal, exposure)
                )
                image = np.asarray(source)
                if image.shape != self._sensor_shape_yx:
                    raise ValueError("virtual frame source returned the wrong shape")
                if image.dtype.kind not in "iu":
                    raise TypeError("virtual frame source must return an integer image")
                x, y, width, height = roi
                image = image[y : y + height, x : x + width]
                image = np.clip(image, 0, np.iinfo(np.uint16).max).astype("<u2", copy=False)
                with self._condition:
                    if stop.is_set() or not self._armed:
                        break
                    self._produced_count += 1
                    record = CameraFrameRecord(
                        image,
                        ordinal,
                        self._produced_count,
                        ordinal,
                        ordinal,
                        None,
                        None,
                        time.time_ns(),
                    )
                    while len(self._queue) >= self._buffer_frame_count:
                        self._queue.popleft()
                    self._queue.append(record)
                    self._condition.notify_all()
        except BaseException as error:
            with self._condition:
                self._worker_error = error
                self._accepting = False
                self._trigger_queue.clear()
                self._condition.notify_all()
        finally:
            with self._condition:
                if self._worker is threading.current_thread():
                    self._worker = None
                    self._worker_stop = None
                self._condition.notify_all()

    def trigger(self, count: int = 1, *, frame: np.ndarray | None = None) -> None:
        count = int(count)
        if count <= 0:
            raise ValueError("trigger count must be positive")
        with self._condition:
            if not self._armed or not self._accepting:
                return
            if self._expected_frames is not None and self._triggered_count + count > self._expected_frames:
                raise RuntimeError("virtual trigger count exceeds the finite arm")
            for _ in range(count):
                self._trigger_queue.append((self._next_ordinal, None if frame is None else np.asarray(frame)))
                self._next_ordinal += 1
                self._triggered_count += 1
            if self._expected_frames is not None and self._triggered_count == self._expected_frames:
                self._accepting = False
            self._condition.notify_all()

    def read_frame_records(self, n: int, *, timeout: float, exact: bool) -> list[CameraFrameRecord]:
        requested = int(n)
        if requested <= 0:
            raise ValueError("n must be positive")
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._condition:
            while len(self._queue) < requested:
                if self._worker_error is not None:
                    raise RuntimeError("virtual camera frame worker failed") from self._worker_error
                worker_running = self._worker is not None and self._worker.is_alive()
                if not self._armed or (not worker_running and not self._trigger_queue and not self._accepting):
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._condition.wait(remaining)
            if exact and len(self._queue) < requested:
                raise TimeoutError("virtual camera did not produce the requested exact frames")
            count = requested if exact else min(requested, len(self._queue))
            records = [self._queue.popleft() for _ in range(count)]
            return records

    def finish_record_capture(self) -> CameraCaptureTerminalRecord:
        with self._condition:
            if self._terminal is not None:
                return self._terminal
            if not self._armed:
                self._terminal = CameraCaptureTerminalRecord(0, True, True, True)
                return self._terminal
            self._accepting = False
            worker = self._worker
            self._condition.notify_all()
        if worker is not None:
            worker.join(timeout=2.0)
        with self._condition:
            if worker is not None and worker.is_alive():
                raise RuntimeError("virtual camera producer did not join")
            if self._worker_error is not None:
                raise RuntimeError("virtual camera frame worker failed") from self._worker_error
            self._armed = False
            self._terminal = CameraCaptureTerminalRecord(
                self._produced_count,
                True,
                not self._queue and not self._trigger_queue,
                True,
            )
            self._condition.notify_all()
            return self._terminal

    def capture_state(self) -> bool:
        with self._condition:
            return self._armed

    def close(self) -> None:
        try:
            if self.capture_state():
                self.finish_record_capture()
        finally:
            with self._condition:
                self._armed = False
                self._accepting = False
                self._queue.clear()
                self._trigger_queue.clear()
                self._condition.notify_all()

    @property
    def produced_count(self) -> int:
        with self._condition:
            return self._produced_count


__all__ = ["VirtualCamera", "VirtualCameraConfig"]
