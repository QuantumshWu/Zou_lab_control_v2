"""Production Hamamatsu DCAM adapter behind the neutral camera contract.

The adapter owns one DCAM handle and one non-daemon SDK thread.  It copies
every accepted driver frame into :class:`CameraFrameRecord` before publishing
it, preserves the DCAM count/stamp/index metadata, and treats an unprovable
ring window as a capture failure.  Hardware identity and one-trigger/one-frame
qualification intentionally remain installation responsibilities.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, replace

import numpy as np

from zlc_data import finite_real, integer, positive_integer


def positive_real(value: object, field: str) -> float:
    return finite_real(value, field, positive=True)


def _snap_roi_axis(
    origin: int,
    extent: int,
    *,
    origin_step: int,
    extent_step: int,
    sensor_extent: int,
) -> tuple[int, int]:
    """The sensor's own grid, snapped so the region asked for is COVERED.

    A subarray can only start and end on the steps the sensor declares, so a
    region drawn on an image is never exactly the region a qCMOS takes.  Which
    way it is rounded is a choice, and rounding the SIZE down was the wrong
    one: the requested region then lost up to a step off its right and bottom
    edges while its origin moved up and left, so a box drawn round the edge
    traps cut them off -- silently, because the applied ROI is what everything
    downstream then measures.

    Selecting a region means covering it.  The origin rounds DOWN to its step
    and the far edge rounds UP to the size step, so the applied subarray
    contains what was asked for, clamped only by the sensor itself.
    """

    origin_step = max(1, int(origin_step))
    extent_step = max(1, int(extent_step))
    sensor_extent = int(sensor_extent)
    start = max(0, min(int(origin), sensor_extent - 1))
    stop = max(start + 1, min(int(origin) + int(extent), sensor_extent))
    snapped_origin = (start // origin_step) * origin_step
    covered = stop - snapped_origin
    snapped_extent = -(-covered // extent_step) * extent_step
    if snapped_origin + snapped_extent > sensor_extent:
        # The far edge cannot be covered without leaving the sensor: take the
        # largest legal subarray that still starts where it should.
        snapped_extent = ((sensor_extent - snapped_origin) // extent_step) * extent_step
    return snapped_origin, max(extent_step, snapped_extent)

from ._dcam_driver import DcamProperty, DcamSdkDriver, DcamValue
from ._owner_lane import CameraSdkOwnerLane
from .contract import (
    CameraCaptureTerminalRecord,
    CameraFrameRecord,
    CameraWorkingPoint,
)
from .photoelectrons import stated_conversion


_CAPTURE_STATUS_BUSY = 1
_WAIT_SLICE_SECONDS = 0.05


class DcamCaptureInterrupted(RuntimeError):
    """A blocked DCAM read was superseded by capture terminalization."""


@dataclass(frozen=True)
class DcamCameraConfig:
    """Requested qCMOS working point.

    ``roi_xywh`` is expressed in unbinned sensor pixels as
    ``(x, y, width, height)``.  Driver-ring cardinality belongs to each arm:
    complete finite cardinality for exact capture, or the declared monitor
    history geometry for continuous acquisition.
    """

    exposure_seconds: float = 0.02
    readout_speed: int = 1
    binning: int = 1
    roi_xywh: tuple[int, int, int, int] | None = None
    device_index: int = 0
    timeout_seconds: float = 10.0
    sensor_mode: int | None = None
    trigger_global_exposure: int | None = None
    #: What one count is worth in photoelectrons, and where zero of them
    #: sits.  A sensor fact, from the datasheet, configured beside the ROI
    #: and the exposure: the ORCA-Quest's ultra-quiet readout is 0.107
    #: electrons per count over an offset of 200.  Both or neither, and
    #: neither means this camera publishes counts and nothing else.
    offset_counts: float | None = None
    electrons_per_count: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "exposure_seconds",
            positive_real(self.exposure_seconds, "exposure_seconds"),
        )
        object.__setattr__(
            self,
            "readout_speed",
            positive_integer(self.readout_speed, "readout_speed"),
        )
        binning = positive_integer(self.binning, "binning")
        if binning not in (1, 2, 4, 8, 16):
            raise ValueError("binning must be one of 1, 2, 4, 8, or 16")
        object.__setattr__(self, "binning", binning)
        device_index = integer(self.device_index, "device_index", minimum=0)
        assert device_index is not None
        object.__setattr__(self, "device_index", device_index)
        object.__setattr__(
            self,
            "timeout_seconds",
            positive_real(self.timeout_seconds, "timeout_seconds"),
        )
        for name in ("sensor_mode", "trigger_global_exposure"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, positive_integer(value, name))
        roi = self.roi_xywh
        if roi is not None:
            if not isinstance(roi, tuple) or len(roi) != 4:
                raise TypeError("roi_xywh must be a four-item tuple or None")
            normalized: list[int] = []
            for index, value in enumerate(roi):
                minimum = 0 if index < 2 else 1
                item = integer(value, f"roi_xywh[{index}]", minimum=minimum)
                assert item is not None
                normalized.append(item)
            object.__setattr__(self, "roi_xywh", tuple(normalized))
        stated_conversion(
            self.offset_counts,
            self.electrons_per_count,
            camera="qCMOS camera",
        )


class DcamCameraAdapter:
    """External-trigger qCMOS implementation of the CameraAdapter protocol."""

    def __init__(
        self,
        config: DcamCameraConfig,
        *,
        driver: object | None = None,
        library_path: str | None = None,
    ) -> None:
        if not isinstance(config, DcamCameraConfig):
            raise TypeError("config must be DcamCameraConfig")
        if driver is not None and library_path is not None:
            raise ValueError("library_path cannot accompany an injected driver")
        self._config = config
        self._driver = (
            DcamSdkDriver(library_path=library_path) if driver is None else driver
        )
        self._lane = CameraSdkOwnerLane("zlc-dcam-camera-owner")
        self._state_lock = threading.RLock()
        self._finish_requested = threading.Event()
        self._device: object | None = None
        self._open = False
        self._armed = False
        self._capture_running = False
        self._ring_size = 0
        self._expected_frames: int | None = 0
        self._copied_count = 0
        self._last_transfer_count = 0
        self._last_transfer_newest: int | None = None
        self._pending: deque[CameraFrameRecord] = deque()
        self._terminal: CameraCaptureTerminalRecord | None = None
        self._terminal_failure: RuntimeError | None = None
        try:
            self._lane.call(self._open_on_owner)
        except BaseException:
            self._lane.close()
            raise

    @property
    def timeout(self) -> float:
        return self._config.timeout_seconds

    @property
    def photoelectron_conversion(self) -> tuple[float, float] | None:
        return stated_conversion(
            self._config.offset_counts,
            self._config.electrons_per_count,
            camera="qCMOS camera",
        )

    def _require_device(self) -> object:
        if self._device is None or not self._open:
            raise RuntimeError("DCAM camera is closed")
        return self._device

    def _open_on_owner(self) -> None:
        # The runtime is the process's and outlives this camera; opening only
        # makes sure it is up.  Taking it down on the way out is what let one
        # camera's close pull it from under a scan, or another camera.
        self._driver.initialize()
        try:
            self._device = self._driver.open_device(self._config.device_index)
            self._open = True
            self._apply_settings_on_owner(self._config)
        except BaseException as primary:
            device = self._device
            self._device = None
            self._open = False
            if device is not None:
                try:
                    device.close()
                except BaseException as secondary:
                    primary.add_note(f"DCAM close after open failure also failed: {secondary}")
            raise

    @staticmethod
    def _integral(value: float, field: str, *, positive: bool = False) -> int:
        result = finite_real(value, field)
        if not result.is_integer():
            raise RuntimeError(f"DCAM returned non-integral {field}: {result!r}")
        normalized = int(result)
        if positive and normalized <= 0:
            raise RuntimeError(f"DCAM returned non-positive {field}: {normalized!r}")
        return normalized

    @staticmethod
    def _set_exact(device: object, property_id: DcamProperty, value: int) -> None:
        applied = float(device.set_get_property(property_id, value))
        if not math.isclose(applied, float(value), rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError(
                f"DCAM property {property_id.name} read back {applied!r}, "
                f"expected {value!r}"
            )

    def _sensor_grid_on_owner(self) -> tuple[int, int, int, int, int, int]:
        """The grid the sensor declares: its size, and the steps it moves in.

        Position and size are separate properties with separate steps -- a
        sensor may place a subarray on a coarser grid than it sizes one -- so
        each is asked for its own rather than one standing in for both.
        """

        device = self._require_device()
        h_step, h_max = device.property_attributes(DcamProperty.SUBARRAY_HSIZE)
        v_step, v_max = device.property_attributes(DcamProperty.SUBARRAY_VSIZE)
        h_pos_step, _h_pos_max = device.property_attributes(DcamProperty.SUBARRAY_HPOS)
        v_pos_step, _v_pos_max = device.property_attributes(DcamProperty.SUBARRAY_VPOS)
        width = self._integral(h_max, "sensor width", positive=True)
        height = self._integral(v_max, "sensor height", positive=True)
        return (
            self._integral(h_pos_step, "horizontal ROI origin step", positive=True),
            self._integral(v_pos_step, "vertical ROI origin step", positive=True),
            self._integral(h_step, "horizontal ROI size step", positive=True),
            self._integral(v_step, "vertical ROI size step", positive=True),
            width,
            height,
        )

    def _apply_settings_on_owner(self, config: DcamCameraConfig) -> CameraWorkingPoint:
        if self._armed:
            raise RuntimeError("qCMOS settings cannot change while armed")
        device = self._require_device()
        device.set_get_property(DcamProperty.EXPOSURE_TIME, config.exposure_seconds)
        self._set_exact(
            device,
            DcamProperty.TRIGGER_SOURCE,
            int(DcamValue.TRIGGER_SOURCE_EXTERNAL),
        )
        self._set_exact(
            device,
            DcamProperty.TRIGGER_ACTIVE,
            int(DcamValue.TRIGGER_ACTIVE_EDGE),
        )
        self._set_exact(
            device,
            DcamProperty.TRIGGER_POLARITY,
            int(DcamValue.TRIGGER_POLARITY_POSITIVE),
        )
        self._set_exact(device, DcamProperty.READOUT_SPEED, config.readout_speed)
        self._set_exact(device, DcamProperty.BINNING, config.binning)
        if config.sensor_mode is not None:
            self._set_exact(device, DcamProperty.SENSOR_MODE, config.sensor_mode)
        if config.trigger_global_exposure is not None:
            self._set_exact(
                device,
                DcamProperty.TRIGGER_GLOBAL_EXPOSURE,
                config.trigger_global_exposure,
            )

        self._set_exact(device, DcamProperty.SUBARRAY_MODE, int(DcamValue.MODE_OFF))
        (
            h_origin_step,
            v_origin_step,
            h_size_step,
            v_size_step,
            sensor_width,
            sensor_height,
        ) = self._sensor_grid_on_owner()
        roi = config.roi_xywh
        if roi is not None:
            requested = tuple(int(value) for value in roi)
            x, y, width, height = requested
            x, width = _snap_roi_axis(
                x,
                width,
                origin_step=h_origin_step,
                extent_step=h_size_step,
                sensor_extent=sensor_width,
            )
            y, height = _snap_roi_axis(
                y,
                height,
                origin_step=v_origin_step,
                extent_step=v_size_step,
                sensor_extent=sensor_height,
            )
            self._set_exact(device, DcamProperty.SUBARRAY_HPOS, 0)
            self._set_exact(device, DcamProperty.SUBARRAY_VPOS, 0)
            self._set_exact(device, DcamProperty.SUBARRAY_HSIZE, width)
            self._set_exact(device, DcamProperty.SUBARRAY_VSIZE, height)
            self._set_exact(device, DcamProperty.SUBARRAY_HPOS, x)
            self._set_exact(device, DcamProperty.SUBARRAY_VPOS, y)
            self._set_exact(device, DcamProperty.SUBARRAY_MODE, int(DcamValue.MODE_ON))
        return self._read_working_point_on_owner()

    def _read_working_point_on_owner(self) -> CameraWorkingPoint:
        device = self._require_device()

        def prop(property_id: DcamProperty) -> float:
            return float(device.get_property(property_id))

        exposure = positive_real(prop(DcamProperty.EXPOSURE_TIME), "applied exposure")
        interval = positive_real(
            prop(DcamProperty.TIMING_MIN_TRIGGER_INTERVAL),
            "minimum external trigger interval",
        )
        integration_offset = finite_real(
            prop(DcamProperty.TIMING_GLOBAL_EXPOSURE_DELAY),
            "global exposure delay",
            minimum=0.0,
        )
        trigger_source = self._integral(prop(DcamProperty.TRIGGER_SOURCE), "trigger source")
        trigger_active = self._integral(prop(DcamProperty.TRIGGER_ACTIVE), "trigger active")
        trigger_polarity = self._integral(
            prop(DcamProperty.TRIGGER_POLARITY), "trigger polarity"
        )
        expected_triggers = (
            int(DcamValue.TRIGGER_SOURCE_EXTERNAL),
            int(DcamValue.TRIGGER_ACTIVE_EDGE),
            int(DcamValue.TRIGGER_POLARITY_POSITIVE),
        )
        if (trigger_source, trigger_active, trigger_polarity) != expected_triggers:
            raise RuntimeError("qCMOS trigger readback is not external rising-edge mode")
        readout_speed = self._integral(
            prop(DcamProperty.READOUT_SPEED), "readout speed", positive=True
        )
        sensor_mode = self._integral(
            prop(DcamProperty.SENSOR_MODE), "sensor mode", positive=True
        )
        global_exposure = self._integral(
            prop(DcamProperty.TRIGGER_GLOBAL_EXPOSURE),
            "trigger global exposure",
            positive=True,
        )
        binning = self._integral(prop(DcamProperty.BINNING), "binning", positive=True)
        if binning not in (1, 2, 4, 8, 16):
            raise RuntimeError(f"qCMOS returned unsupported binning {binning}")
        *_steps, sensor_width, sensor_height = self._sensor_grid_on_owner()
        subarray_mode = self._integral(
            prop(DcamProperty.SUBARRAY_MODE), "subarray mode", positive=True
        )
        if subarray_mode == int(DcamValue.MODE_OFF):
            origin_yx = (0, 0)
            roi_shape_yx = (sensor_height, sensor_width)
        elif subarray_mode == int(DcamValue.MODE_ON):
            x = self._integral(prop(DcamProperty.SUBARRAY_HPOS), "subarray x")
            y = self._integral(prop(DcamProperty.SUBARRAY_VPOS), "subarray y")
            width = self._integral(
                prop(DcamProperty.SUBARRAY_HSIZE), "subarray width", positive=True
            )
            height = self._integral(
                prop(DcamProperty.SUBARRAY_VSIZE), "subarray height", positive=True
            )
            if x < 0 or y < 0 or x + width > sensor_width or y + height > sensor_height:
                raise RuntimeError("qCMOS returned an invalid subarray readback")
            origin_yx = (y, x)
            roi_shape_yx = (height, width)
        else:
            raise RuntimeError(f"qCMOS returned unsupported subarray mode {subarray_mode}")
        frame_width = self._integral(
            prop(DcamProperty.IMAGE_WIDTH), "image width", positive=True
        )
        frame_height = self._integral(
            prop(DcamProperty.IMAGE_HEIGHT), "image height", positive=True
        )
        pixel_type = self._integral(
            prop(DcamProperty.IMAGE_PIXEL_TYPE), "image pixel type", positive=True
        )
        if pixel_type == int(DcamValue.PIXEL_MONO16):
            dtype = np.dtype("<u2")
        elif pixel_type == int(DcamValue.PIXEL_MONO8):
            dtype = np.dtype("u1")
        else:
            raise RuntimeError(f"qCMOS returned unsupported pixel type {pixel_type}")
        frame_shape_yx = (frame_height, frame_width)
        if frame_shape_yx != (
            roi_shape_yx[0] // binning,
            roi_shape_yx[1] // binning,
        ):
            raise RuntimeError(
                "qCMOS image geometry differs from ROI/binning readback"
            )
        conversion = self.photoelectron_conversion
        return CameraWorkingPoint(
            acquisition_mode="EXTERNAL_TRIGGERED",
            frame_shape_yx=frame_shape_yx,
            sensor_shape_yx=(sensor_height, sensor_width),
            roi_origin_yx=origin_yx,
            roi_shape_yx=roi_shape_yx,
            binning_yx=(binning, binning),
            dtype=dtype,
            count_unit="count",
            exposure_seconds=exposure,
            required_external_trigger_interval_seconds=interval,
            external_trigger_integration_start_offset_seconds=integration_offset,
            gain=1.0,
            readout_mode=(
                "dcam:"
                f"readout={readout_speed};sensor={sensor_mode};"
                f"global_exposure={global_exposure}"
            ),
            offset_counts=None if conversion is None else conversion[0],
            electrons_per_count=None if conversion is None else conversion[1],
        )

    def working_point(self) -> CameraWorkingPoint:
        return self._lane.call(self._read_working_point_on_owner)

    def set_exposure_seconds(self, seconds: float) -> CameraWorkingPoint:
        """Integrate for this long on every trigger, leaving the geometry."""

        return self._reconfigure(
            replace(
                self._config,
                exposure_seconds=positive_real(seconds, "exposure_seconds"),
            )
        )

    def set_roi(
        self, roi_xywh: tuple[int, int, int, int] | None
    ) -> CameraWorkingPoint:
        """Read this part of the sensor, leaving the exposure.

        ``None`` is the whole sensor, which is what an operator means by no
        ROI at all.
        """

        return self._reconfigure(replace(self._config, roi_xywh=roi_xywh))

    def _reconfigure(self, candidate: "DcamCameraConfig") -> CameraWorkingPoint:
        """Apply one changed field and keep the config at what the sensor did."""

        def apply() -> CameraWorkingPoint:
            point = self._apply_settings_on_owner(candidate)
            actual_roi = None
            if candidate.roi_xywh is not None:
                actual_roi = (
                    point.roi_origin_yx[1],
                    point.roi_origin_yx[0],
                    point.roi_shape_yx[1],
                    point.roi_shape_yx[0],
                )
            self._config = replace(
                candidate,
                exposure_seconds=point.exposure_seconds,
                roi_xywh=actual_roi,
            )
            return point

        return self._lane.call(apply)

    @staticmethod
    def _finite_groups(
        frames: int | None,
        source_group_sizes: tuple[int, ...] | None,
    ) -> tuple[int | None, tuple[int, ...] | None]:
        if frames is None:
            groups = tuple(int(value) for value in (source_group_sizes or ()))
            if groups and (len(groups) != 1 or groups[0] <= 0):
                raise ValueError(
                    "continuous external capture requires one positive source group"
                )
            return None, groups or None
        expected = positive_integer(frames, "frames")
        if not isinstance(source_group_sizes, tuple):
            raise TypeError("finite arm requires tuple source_group_sizes")
        groups = tuple(
            positive_integer(value, "source_group_sizes item")
            for value in source_group_sizes
        )
        if not groups or sum(groups) != expected:
            raise ValueError("source_group_sizes must exactly cover frames")
        return expected, groups

    def arm(
        self,
        frames: int | None,
        *,
        source_group_sizes: tuple[int, ...] | None,
        buffer_frame_count: int,
        timeout: float,
    ) -> None:
        expected, _groups = self._finite_groups(frames, source_group_sizes)
        buffer_count = positive_integer(buffer_frame_count, "buffer_frame_count")
        positive_real(timeout, "timeout")
        if expected is not None and buffer_count != expected:
            raise ValueError(
                "finite buffer_frame_count must equal the complete frame count"
            )

        def arm_on_owner() -> None:
            if self._armed:
                raise RuntimeError("qCMOS is already armed")
            device = self._require_device()
            before = self._read_working_point_on_owner()
            device.allocate_buffer(buffer_count)
            started = False
            try:
                device.start_capture()
                started = True
                self._capture_running = True
                after = self._read_working_point_on_owner()
                if after != before:
                    raise RuntimeError("qCMOS working point changed across arm")
                count, newest = device.transfer_info()
                if int(count) != 0 or int(newest) != -1:
                    raise RuntimeError(
                        "qCMOS transfer counter did not reset at the arm boundary"
                    )
            except BaseException as primary:
                stopped = not started
                if started:
                    try:
                        device.stop_capture()
                    except BaseException as secondary:
                        primary.add_note(f"DCAM stop after arm failure also failed: {secondary}")
                    else:
                        stopped = True
                        self._capture_running = False
                if stopped:
                    try:
                        device.release_buffer()
                    except BaseException as secondary:
                        primary.add_note(
                            f"DCAM release after arm failure also failed: {secondary}"
                        )
                        stopped = False
                if not stopped:
                    # The driver still owns storage or may still be capturing.
                    # Preserve that fact so explicit terminalization can retry;
                    # never hide it behind a failed arm result.
                    with self._state_lock:
                        self._finish_requested.set()
                        self._armed = True
                        self._ring_size = buffer_count
                        self._expected_frames = expected
                        self._copied_count = 0
                        self._last_transfer_count = 0
                        self._last_transfer_newest = None
                        self._pending.clear()
                        self._terminal = None
                raise
            with self._state_lock:
                self._finish_requested.clear()
                self._armed = True
                self._capture_running = True
                self._ring_size = buffer_count
                self._expected_frames = expected
                self._copied_count = 0
                self._last_transfer_count = 0
                self._last_transfer_newest = None
                self._pending.clear()
                self._terminal = None
                self._terminal_failure = None

        self._lane.call(arm_on_owner)

    def _check_not_finishing(self) -> None:
        if self._finish_requested.is_set():
            raise DcamCaptureInterrupted("qCMOS capture is terminalizing")

    def _observe_transfer_on_owner(self) -> tuple[int, int | None]:
        device = self._require_device()
        count_raw, newest_raw = device.transfer_info()
        count = int(count_raw)
        newest = int(newest_raw)
        if count < 0:
            raise RuntimeError("qCMOS produced count is negative")
        if count < self._last_transfer_count or count < self._copied_count:
            raise RuntimeError("qCMOS produced count moved backwards")
        if count == 0:
            if newest != -1:
                raise RuntimeError("qCMOS empty transfer window has a newest frame index")
            newest_value: int | None = None
        else:
            if not 0 <= newest < self._ring_size:
                raise RuntimeError("qCMOS newest frame index is outside the allocated ring")
            newest_value = newest
            if self._last_transfer_count > 0 and self._last_transfer_newest is not None:
                expected_newest = (
                    self._last_transfer_newest + count - self._last_transfer_count
                ) % self._ring_size
                if newest != expected_newest:
                    raise RuntimeError(
                        "qCMOS transfer count/newest-index pair is inconsistent"
                    )
        if self._expected_frames is not None and count > self._expected_frames:
            raise RuntimeError("qCMOS produced more frames than the finite arm declared")
        self._last_transfer_count = count
        self._last_transfer_newest = newest_value
        return count, newest_value

    def _drain_snapshot_on_owner(
        self,
        available: int,
        newest: int,
    ) -> None:
        """Copy out the frames the sensor has ALREADY produced.

        No deadline governs this.  A read's timeout says how long to wait for
        a frame; it does not say how long a frame that has already arrived may
        take to copy -- that is work which must finish, because the ring will
        overwrite what is not copied out of it.  Enforcing the read deadline
        here made the two indistinguishable, and a frame arriving near the end
        of a read window was then reported as a timeout INSTEAD of being
        returned: with a 50 ms monitor slice and a several-hundred-ms cycle,
        that is every cycle, which is how "the deadline expired before frame
        commit" came to greet a measurement that had just started.

        The work is bounded without a clock: the snapshot names how many
        frames to copy, and the loop stops there.  Cancellation still lands
        immediately, through the finishing checks between copies.
        """

        if available - self._copied_count > self._ring_size:
            raise RuntimeError("qCMOS ring overrun overwrote an unread frame")
        device = self._require_device()
        snapshot_available = available
        snapshot_newest = newest
        while self._copied_count < snapshot_available:
            self._check_not_finishing()
            ordinal = self._copied_count
            distance = snapshot_available - 1 - ordinal
            ring_index = (snapshot_newest - distance) % self._ring_size
            image, frame_stamp, camera_stamp, seconds, microseconds = device.copy_frame(
                ring_index
            )
            self._check_not_finishing()
            post_count, _post_newest = self._observe_transfer_on_owner()
            if post_count - ordinal > self._ring_size:
                raise RuntimeError(
                    "qCMOS ring advanced far enough to overwrite a frame during copy"
                )
            record = CameraFrameRecord(
                image=image,
                source_ordinal=ordinal,
                produced_count=ordinal + 1,
                frame_stamp=frame_stamp,
                camera_stamp=camera_stamp,
                timestamp_seconds=seconds,
                timestamp_microseconds=microseconds,
                host_received_at_ns=time.time_ns(),
                driver_buffer_index=ring_index,
            )
            self._check_not_finishing()
            with self._state_lock:
                self._pending.append(record)
                self._copied_count += 1

    def _read_on_owner(
        self,
        wanted: int,
        timeout: float,
        exact: bool,
    ) -> list[CameraFrameRecord]:
        if not self._armed:
            raise RuntimeError("qCMOS read requires an armed capture")
        deadline = time.monotonic() + timeout
        result: list[CameraFrameRecord] = []
        device = self._require_device()
        while len(result) < wanted:
            self._check_not_finishing()
            with self._state_lock:
                while self._pending and len(result) < wanted:
                    result.append(self._pending.popleft())
            if len(result) == wanted:
                break
            available, newest = self._observe_transfer_on_owner()
            if available > self._copied_count:
                assert newest is not None
                self._drain_snapshot_on_owner(available, newest)
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            ready = device.wait_frame_ready(
                max(1, int(min(_WAIT_SLICE_SECONDS, remaining) * 1000.0))
            )
            self._check_not_finishing()
            if not ready and time.monotonic() >= deadline:
                break
        if len(result) != wanted and exact:
            raise TimeoutError(
                f"qCMOS returned {len(result)} of {wanted} exact frame(s)"
            )
        return result

    def read_frame_records(
        self,
        n: int,
        *,
        timeout: float,
        exact: bool,
    ) -> list[CameraFrameRecord]:
        wanted = positive_integer(n, "n")
        bounded_timeout = finite_real(timeout, "timeout", minimum=0.0)
        if type(exact) is not bool:
            raise TypeError("exact must be bool")
        return self._lane.call(
            lambda: self._read_on_owner(wanted, bounded_timeout, exact)
        )

    def _finish_on_owner(self) -> CameraCaptureTerminalRecord:
        if self._terminal is not None:
            return self._terminal
        prior_terminal_failure = self._terminal_failure
        if not self._armed:
            if prior_terminal_failure is not None:
                raise prior_terminal_failure
            self._terminal = CameraCaptureTerminalRecord(0, True, True, True)
            return self._terminal
        device = self._require_device()
        if self._capture_running:
            device.stop_capture()
            if int(device.capture_status()) == _CAPTURE_STATUS_BUSY:
                raise RuntimeError("qCMOS remained busy after cap_stop; ring retained")
            self._capture_running = False

        def stable_snapshot() -> tuple[int, int]:
            count_raw, newest_raw = device.transfer_info()
            count = int(count_raw)
            newest = int(newest_raw)
            if count < max(
                self._copied_count,
                self._last_transfer_count,
            ):
                raise RuntimeError("qCMOS final produced count moved backwards")
            if count == 0:
                if newest != -1:
                    raise RuntimeError("qCMOS final empty window has a newest index")
            elif not 0 <= newest < self._ring_size:
                raise RuntimeError("qCMOS final newest index is outside the ring")
            return count, newest

        final_error: BaseException | None = None
        produced_count = 0
        try:
            first = stable_snapshot()
            second = stable_snapshot()
            if first != second:
                raise RuntimeError("qCMOS transfer state did not stabilize after cap_stop")
            produced_count, newest = second
            if self._last_transfer_count > 0 and self._last_transfer_newest is not None:
                expected_newest = (
                    self._last_transfer_newest
                    + produced_count
                    - self._last_transfer_count
                ) % self._ring_size
                if newest != expected_newest:
                    raise RuntimeError(
                        "qCMOS final count/newest-index pair is inconsistent"
                    )
        except BaseException as error:
            final_error = error
        try:
            device.release_buffer()
        except BaseException:
            raise RuntimeError("qCMOS buffer release failed after cap_stop")
        with self._state_lock:
            self._armed = False
            self._capture_running = False
            self._ring_size = 0
            self._expected_frames = 0
            self._pending.clear()
        if final_error is not None:
            if prior_terminal_failure is not None:
                raise prior_terminal_failure
            terminal_failure = RuntimeError(
                "qCMOS final transfer state is not authoritative"
            )
            self._terminal_failure = terminal_failure
            raise terminal_failure from final_error
        if prior_terminal_failure is not None:
            raise prior_terminal_failure
        self._terminal = CameraCaptureTerminalRecord(
            produced_count,
            True,
            True,
            True,
        )
        return self._terminal

    def finish_record_capture(self) -> CameraCaptureTerminalRecord:
        self._finish_requested.set()
        return self._lane.call(self._finish_on_owner)

    def capture_state(self) -> bool:
        with self._state_lock:
            return self._armed

    def observed_produced_count(self) -> int:
        """Read the DCAM transfer counter without consuming a frame."""

        def observe() -> int:
            if not self._armed:
                raise RuntimeError("qCMOS produced count requires an armed capture")
            count, _newest = self._observe_transfer_on_owner()
            return count

        return self._lane.call(observe)

    def _close_on_owner(self) -> None:
        """Release the handle and the runtime, whichever step fails.

        A driver that refuses one of them must not leave the other held: the
        next open then finds a device already in use, from a process that
        thinks it closed.
        """

        device = self._device
        try:
            if device is not None:
                device.close()
        finally:
            self._device = None
            self._open = False

    def close(self) -> None:
        """Stop the capture, release the handle, retire the lane -- all of them.

        This was a bare statement sequence, and finish_record_capture raises on
        several ordinary hardware conditions.  When it did, the SDK handle was
        abandoned and its non-daemon lane thread was left running: the process
        could not exit, and the camera stayed claimed until it was power-cycled.
        """

        if not self._open and self._device is None:
            self._lane.close()
            return
        failures: list[BaseException] = []
        try:
            if self.capture_state():
                self.finish_record_capture()
        except BaseException as error:  # noqa: BLE001 - the handle still goes
            failures.append(error)
        try:
            self._lane.call(self._close_on_owner)
        except BaseException as error:  # noqa: BLE001 - the lane still retires
            failures.append(error)
        finally:
            self._lane.close()
        if failures:
            raise failures[0]


__all__ = [
    "DcamCameraAdapter",
    "DcamCameraConfig",
    "DcamCaptureInterrupted",
]
