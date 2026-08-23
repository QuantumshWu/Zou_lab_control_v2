"""Virtual camera adapter with asynchronous bounded-buffer semantics."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Callable
from uuid import uuid4

import numpy as np

from zlc_atom.authoring import AuthoringField, TunableField
from zlc_atom.devices.camera.contract import (
    CameraCaptureTerminalRecord,
    CameraFrameRecord,
    CameraWorkingPoint,
)
from zlc_atom.devices.camera.photoelectrons import stated_conversion
from .world import DEFAULT_SIMULATION_IMAGE_SHAPE_YX


@dataclass(frozen=True)
class VirtualCameraConfig:
    """Geometry, exposure and pixel format; imaging physics belongs to SimulationWorld.

    ``frame_dtype`` states the sensor's native unsigned pixel format, exactly
    as the real adapters do: the qCMOS site camera reads out Mono16 (``<u2``)
    and the Basler MOT monitor reads out Mono8 (``|u1``).
    """

    frame_shape_yx: tuple[int, int] = DEFAULT_SIMULATION_IMAGE_SHAPE_YX
    exposure_seconds: float = 0.02
    frame_dtype: str = "<u2"
    #: What one count is worth, as this sensor's own world applies it going
    #: the other way.  ``None`` for a sensor that states no conversion, which
    #: is how the MOT monitor stands in for a machine-vision camera.
    offset_counts: float | None = None
    electrons_per_count: float | None = None


class VirtualCamera:
    """A trigger-driven camera whose frame source is injected by the world."""

    def __init__(
        self,
        config: VirtualCameraConfig | None = None,
        *,
        frame_source: Callable[[int, float], np.ndarray] | None = None,
        free_running: bool = False,
    ) -> None:
        if frame_source is None or not callable(frame_source):
            raise TypeError("virtual camera requires an injected frame_source")
        self.config = VirtualCameraConfig() if config is None else config
        shape = tuple(int(item) for item in self.config.frame_shape_yx)
        if len(shape) != 2 or any(item <= 0 for item in shape):
            raise ValueError("frame_shape_yx must contain two positive dimensions")
        if float(self.config.exposure_seconds) <= 0:
            raise ValueError("exposure_seconds must be positive")
        dtype = np.dtype(self.config.frame_dtype)
        if dtype.kind != "u":
            raise ValueError("frame_dtype must be an unsigned integer dtype")
        self._frame_dtype = dtype
        self._frame_limits = np.iinfo(dtype)
        #: Reused per-frame clip target; CameraFrameRecord makes the one
        #: immutable bytes copy, so reuse never aliases a published frame.
        self._clip_buffer: np.ndarray | None = None
        self._frame_source = frame_source
        self._free_running = bool(free_running)
        self._condition = threading.Condition()
        self._device_session_id = uuid4().hex
        self._settings_epoch = 0
        self._sensor_shape_yx = shape
        self._exposure_seconds = float(self.config.exposure_seconds)
        self._roi_xywh = (0, 0, shape[1], shape[0])
        self._queue: deque[CameraFrameRecord] = deque()
        self._trigger_queue: deque[
            tuple[
                int,
                np.ndarray | None,
                float,
                tuple[int, int, int, int],
                str,
                int,
            ]
        ] = deque()
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
        return self._frame_dtype

    @property
    def photoelectron_conversion(self) -> tuple[float, float] | None:
        return stated_conversion(
            self.config.offset_counts,
            self.config.electrons_per_count,
            camera="virtual camera",
        )

    def working_point(self) -> CameraWorkingPoint:
        with self._condition:
            exposure = self._exposure_seconds
            x, y, width, height = self._roi_xywh
        return CameraWorkingPoint(
            "FREE_RUNNING" if self._free_running else "EXTERNAL_TRIGGERED",
            (height, width),
            self._sensor_shape_yx,
            (y, x),
            (height, width),
            (1, 1),
            self.frame_dtype,
            "count",
            exposure,
            None if self._free_running else exposure,
            None if self._free_running else 0.0,
            1.0,
            (
                "virtual-free-running"
                if self._free_running
                else "virtual-external-trigger"
            ),
            self.config.offset_counts,
            self.config.electrons_per_count,
        )

    def set_exposure_seconds(self, seconds: float) -> CameraWorkingPoint:
        """Integrate for this long on every trigger, leaving the geometry."""

        exposure = float(seconds)
        if not np.isfinite(exposure) or exposure <= 0:
            raise ValueError("exposure_seconds must be positive and finite")
        with self._condition:
            if self._armed:
                raise RuntimeError("virtual camera settings cannot change while armed")
            if self._exposure_seconds != exposure:
                self._exposure_seconds = exposure
                self._settings_epoch += 1
        return self.working_point()

    def set_roi(
        self, roi_xywh: tuple[int, int, int, int] | None
    ) -> CameraWorkingPoint:
        """Read this part of the sensor, leaving the exposure.

        ``None`` is the whole sensor, which is what an operator means by no
        ROI at all.
        """

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
            self._roi_xywh = roi
        return self.working_point()

    def tunable_fields(self) -> tuple[TunableField, ...]:
        """The runtime knobs this camera volunteers to a scan (duck-typed).

        Both bounds are declared because a scan plan must be refusable
        against a finite range; the worker reads the live value per frame,
        so tuning works while armed -- which is exactly when a scan needs it.
        """

        with self._condition:
            current = float(self._exposure_seconds)
        return (
            TunableField(
                metadata=AuthoringField(
                    "exposure_seconds",
                    "float",
                    "Exposure (s)",
                    float(self.config.exposure_seconds),
                    minimum=1e-6,
                    maximum=10.0,
                ),
                current=current,
                live_write=True,
                dependency_group=("exposure_seconds",),
            ),
        )

    def tunable_values(self) -> dict[str, float]:
        with self._condition:
            return {"exposure_seconds": float(self._exposure_seconds)}

    def settings_provenance(self) -> dict[str, object]:
        with self._condition:
            return {
                "device_session_id": self._device_session_id,
                "settings_epoch": self._settings_epoch,
            }

    def tune(self, name: str, value: float) -> float:
        (tunable,) = self.tunable_fields()
        field = tunable.metadata
        if str(name) != field.name:
            raise ValueError(
                f"virtual camera has no tunable field {name!r}; "
                f"it offers {field.name!r}"
            )
        exposure = float(value)
        if not np.isfinite(exposure) or not (field.minimum <= exposure <= field.maximum):
            raise ValueError(
                f"exposure_seconds must lie in [{field.minimum:g}, {field.maximum:g}]"
            )
        with self._condition:
            if exposure != self._exposure_seconds:
                self._exposure_seconds = exposure
                self._settings_epoch += 1
                self._condition.notify_all()
            return float(self._exposure_seconds)

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
                    if self._free_running:
                        if not self._accepting or stop.is_set():
                            break
                        exposure = self._exposure_seconds
                        roi = self._roi_xywh
                        settings_session_id = self._device_session_id
                        settings_epoch = self._settings_epoch
                        deadline = time.monotonic() + exposure
                        while self._accepting and not stop.is_set():
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                break
                            self._condition.wait(timeout=remaining)
                        if not self._accepting or stop.is_set():
                            break
                        ordinal = self._next_ordinal
                        self._next_ordinal += 1
                        provided = None
                    else:
                        while (
                            not self._trigger_queue
                            and self._accepting
                            and not stop.is_set()
                        ):
                            self._condition.wait()
                        if self._trigger_queue:
                            (
                                ordinal,
                                provided,
                                exposure,
                                roi,
                                settings_session_id,
                                settings_epoch,
                            ) = self._trigger_queue.popleft()
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
                # One in-place clip into a reused buffer instead of a fresh
                # clip+astype allocation per frame; CameraFrameRecord then
                # snapshots the buffer into immutable bytes -- the single
                # per-frame copy, made here on the producer thread, which
                # every downstream freeze retains as a view.
                buffer = self._clip_buffer
                if buffer is None or buffer.shape != image.shape:
                    buffer = np.empty(image.shape, dtype=self._frame_dtype)
                    self._clip_buffer = buffer
                np.clip(
                    image,
                    self._frame_limits.min,
                    self._frame_limits.max,
                    out=buffer,
                    casting="unsafe",
                )
                with self._condition:
                    if stop.is_set() or not self._armed:
                        break
                    self._produced_count += 1
                    if (
                        self._expected_frames is not None
                        and self._produced_count >= self._expected_frames
                    ):
                        self._accepting = False
                    record = CameraFrameRecord(
                        buffer,
                        ordinal,
                        self._produced_count,
                        ordinal,
                        ordinal,
                        None,
                        None,
                        time.time_ns(),
                        settings_session_id=settings_session_id,
                        settings_epochs=(settings_epoch,),
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

    def trigger(
        self,
        count: int = 1,
        *,
        frame: np.ndarray | None = None,
    ) -> None:
        if self._free_running:
            raise RuntimeError("free-running virtual camera does not accept triggers")
        count = int(count)
        if count <= 0:
            raise ValueError("trigger count must be positive")
        with self._condition:
            if not self._armed or not self._accepting:
                return
            if (
                self._expected_frames is not None
                and self._triggered_count + count > self._expected_frames
            ):
                raise RuntimeError("virtual trigger count exceeds the finite arm")
            for _ in range(count):
                ordinal = self._next_ordinal
                self._next_ordinal += 1
                self._trigger_queue.append(
                    (
                        ordinal,
                        None if frame is None else np.asarray(frame),
                        self._exposure_seconds,
                        self._roi_xywh,
                        self._device_session_id,
                        self._settings_epoch,
                    )
                )
                self._triggered_count += 1
                if (
                    self._expected_frames is not None
                    and self._triggered_count == self._expected_frames
                ):
                    self._accepting = False
                    break
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
