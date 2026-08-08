"""Basler pylon camera: the MOT viewer, as a pure frame grabber.

Ported from the pre-split tree against this package's ``CameraAdapter``
contract.  Not a byte-copy -- the old driver was written against a different
base class -- but every behaviour that was learned the hard way is carried over,
and each is commented where it lives.

The camera never touches a sequencer.  Whatever gates its frames is somebody
else's business, which is what makes the virtual twin and this a drop-in swap.

``pypylon`` is imported lazily, so a machine with no Basler runtime still
imports this package, runs the virtual backend, and passes the whole suite.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Sequence

import numpy as np

from .contract import (
    CameraAcquisitionMode,
    CameraCaptureTerminalRecord,
    CameraFrameRecord,
    CameraWorkingPoint,
)


__all__ = ["PylonCameraAdapter", "PylonCameraConfig"]


def _snap_to_increment(value: int, low: int, high: int, increment: int) -> int:
    """Clamp into the camera's own range and align to its GenICam increment.

    The hardware owns the legal grid, not a constant in our source: a value the
    sensor cannot take is rejected outright, and one silently rounded elsewhere
    would make the working point we record a lie.
    """

    increment = max(1, int(increment))
    clamped = max(int(low), min(int(high), int(value)))
    return int(low) + ((clamped - int(low)) // increment) * increment


def _roi_request(
    value: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise TypeError("roi_xywh must contain four integers or be None") from exc
    if len(result) != 4:
        raise ValueError("roi_xywh must contain four integers or be None")
    return result


@dataclass(frozen=True)
class PylonCameraConfig:
    """What an operator writes down to reach and set up one Basler camera."""

    serial: str = ""
    exposure_seconds: float = 5e-3
    trigger_source: str = "Software"
    pixel_format: str = "Mono8"
    roi_xywh: tuple[int, int, int, int] | None = None
    timeout_seconds: float = 2.0

    @property
    def free_run(self) -> bool:
        """Software trigger means free-run: nothing external gates the frames."""

        return str(self.trigger_source).strip().lower() == "software"


class PylonCameraAdapter:
    """A Basler camera behind the camera contract every backend keeps."""

    def __init__(self, config: PylonCameraConfig, *, camera: object | None = None) -> None:
        self.config = config
        self._camera = camera
        self._armed_total: int | None = None
        self._grabbed = 0
        self._armed = False
        self._roi = _roi_request(config.roi_xywh)
        self._configured = False
        self._capture_incomplete = False

    # ------------------------------------------------------------------ open

    def open(self) -> None:
        """Attach to the camera and push every configured setting to it.

        Attaching and configuring are separate steps on purpose.  Devices are
        opened last, so configure-then-open is the normal order, and a driver
        that only applies settings it receives while already open images the
        full sensor while faithfully reporting the ROI it was handed.

        pypylon is imported only here: a machine with no Basler runtime must
        still import this module and run everything that touches no Basler.
        """

        if self._camera is None:
            self._attach()
        if self._configured:
            return
        self._apply_pixel_format()
        self._apply_trigger()
        self._apply_exposure()
        self._apply_roi()
        self._configured = True

    def _attach(self) -> None:
        from pypylon import pylon  # noqa: PLC0415 -- lazy on purpose, see open()

        factory = pylon.TlFactory.GetInstance()
        if self.config.serial:
            devices = [
                info
                for info in factory.EnumerateDevices()
                if info.GetSerialNumber() == self.config.serial
            ]
            if not devices:
                raise RuntimeError(f"no Basler camera with serial {self.config.serial!r}")
            device = factory.CreateDevice(devices[0])
        else:
            device = factory.CreateFirstDevice()
        self._camera = pylon.InstantCamera(device)
        self._camera.Open()

    def close(self) -> None:
        camera = self._camera
        if camera is None:
            return
        if camera.IsGrabbing():
            camera.StopGrabbing()
        camera.Close()
        self._camera = None
        self._configured = False

    # ------------------------------------------------------------ configuring

    def _paused_stream(self):
        """Stop the grab stream for a setting that cannot change while it runs."""

        camera = self._camera

        class _Pause:
            def __enter__(self_inner):
                self_inner.was_grabbing = camera is not None and camera.IsGrabbing()
                if self_inner.was_grabbing:
                    camera.StopGrabbing()
                return self_inner

            def __exit__(self_inner, *_exc):
                if self_inner.was_grabbing:
                    from pypylon import pylon  # noqa: PLC0415

                    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                return False

        return _Pause()

    def _apply_exposure(self) -> None:
        # Basler exposes ExposureTime in microseconds, and it is legal to change
        # while grabbing -- no stream pause needed.
        self._camera.ExposureTime.SetValue(float(self.config.exposure_seconds) * 1e6)

    def _apply_pixel_format(self) -> None:
        with self._paused_stream():
            self._camera.PixelFormat.SetValue(str(self.config.pixel_format))

    def _apply_trigger(self) -> None:
        camera = self._camera
        camera.TriggerSelector.SetValue("FrameStart")
        if self.config.free_run:
            camera.TriggerMode.SetValue("Off")
        else:
            camera.TriggerMode.SetValue("On")
            camera.TriggerSource.SetValue(str(self.config.trigger_source))

    def _apply_roi(self) -> None:
        """Push the ROI in the GenICam-safe order.

        Zero the offsets, size the window, then place it.  Setting Width while a
        stale OffsetX is still active can violate ``offset + width <= sensor``
        and the camera rejects the write outright -- the same zero-offsets-first
        dance the qCMOS subarray code does.
        """

        camera = self._camera
        if camera is None:
            return
        with self._paused_stream():
            camera.OffsetX.SetValue(int(camera.OffsetX.GetMin()))
            camera.OffsetY.SetValue(int(camera.OffsetY.GetMin()))
            if self._roi is None:
                # Blank means the FULL sensor, never a stale window.
                camera.Width.SetValue(int(camera.WidthMax.GetValue()))
                camera.Height.SetValue(int(camera.HeightMax.GetValue()))
                return
            x, y, width, height = self._roi
            sensor_width = int(camera.WidthMax.GetValue())
            sensor_height = int(camera.HeightMax.GetValue())
            width = _snap_to_increment(
                width,
                camera.Width.GetMin(),
                min(camera.Width.GetMax(), sensor_width),
                camera.Width.GetInc(),
            )
            height = _snap_to_increment(
                height,
                camera.Height.GetMin(),
                min(camera.Height.GetMax(), sensor_height),
                camera.Height.GetInc(),
            )
            camera.Width.SetValue(width)
            camera.Height.SetValue(height)
            x = _snap_to_increment(
                x,
                camera.OffsetX.GetMin(),
                min(camera.OffsetX.GetMax(), sensor_width - width),
                camera.OffsetX.GetInc(),
            )
            y = _snap_to_increment(
                y,
                camera.OffsetY.GetMin(),
                min(camera.OffsetY.GetMax(), sensor_height - height),
                camera.OffsetY.GetInc(),
            )
            camera.OffsetX.SetValue(x)
            camera.OffsetY.SetValue(y)
            # Record what the sensor actually granted, not what we asked for.
            self._roi = (
                int(camera.OffsetX.GetValue()),
                int(camera.OffsetY.GetValue()),
                int(camera.Width.GetValue()),
                int(camera.Height.GetValue()),
            )

    # -------------------------------------------------------------- contract

    @property
    def timeout(self) -> float:
        return float(self.config.timeout_seconds)

    def configure_measurement(
        self,
        *,
        exposure_seconds: float,
        roi_xywh: tuple[int, int, int, int] | None,
    ) -> CameraWorkingPoint:
        exposure = float(exposure_seconds)
        if not np.isfinite(exposure) or exposure <= 0:
            raise ValueError("exposure_seconds must be positive and finite")
        roi = _roi_request(roi_xywh)
        if self._armed:
            raise RuntimeError("pylon settings cannot change while armed")
        self.open()
        candidate = replace(
            self.config,
            exposure_seconds=exposure,
            roi_xywh=roi,
        )
        self.config = candidate
        self._roi = roi
        self._apply_exposure()
        self._apply_roi()
        point = self.capture_working_point()
        actual_roi = None
        if roi is not None:
            actual_roi = (
                point.roi_origin_yx[1],
                point.roi_origin_yx[0],
                point.roi_shape_yx[1],
                point.roi_shape_yx[0],
            )
        self.config = replace(
            candidate,
            exposure_seconds=point.exposure_seconds,
            roi_xywh=actual_roi,
        )
        return point

    def capture_working_point(self) -> CameraWorkingPoint:
        """Read the sensor's state back, rather than repeating what we asked for."""

        self.open()
        camera = self._camera
        width = int(camera.Width.GetValue())
        height = int(camera.Height.GetValue())
        sensor = (int(camera.HeightMax.GetValue()), int(camera.WidthMax.GetValue()))
        origin = (int(camera.OffsetY.GetValue()), int(camera.OffsetX.GetValue()))
        pixel_format = str(camera.PixelFormat.GetValue())
        dtype = np.dtype("uint8") if pixel_format.endswith("8") else np.dtype("uint16")
        exposure = float(camera.ExposureTime.GetValue()) / 1e6
        return CameraWorkingPoint(
            acquisition_mode=(
                CameraAcquisitionMode.FREE_RUNNING
                if self.config.free_run
                else CameraAcquisitionMode.EXTERNAL_TRIGGERED
            ),
            frame_shape_yx=(height, width),
            sensor_shape_yx=sensor,
            roi_origin_yx=origin,
            roi_shape_yx=(height, width),
            binning_yx=(1, 1),
            dtype=dtype,
            count_unit="count",
            exposure_seconds=exposure,
            required_external_trigger_interval_seconds=exposure,
            external_trigger_integration_start_offset_seconds=0.0,
            gain=1.0,
            readout_mode=f"pylon-{pixel_format}-{self.config.trigger_source}",
        )

    def arm(
        self,
        frames: int | None,
        *,
        source_group_sizes: tuple[int, ...] | None,
        buffer_frame_count: int,
        timeout: float,
    ) -> None:
        """Start the grab session, with the strategy each mode's semantics need.

        FREE-RUN keeps the stream RESIDENT: it starts on the first arm and stays
        running across arm/disarm.  Restarting a USB3 stream per frame costs tens
        of milliseconds and was the live-monitor stutter.  Latest-image-only
        means a slow viewer always sees the current image rather than working
        through a stale backlog.

        HARDWARE TRIGGER gets one bounded session per arm, stopped when done:
        one trigger, one frame, one shot, strictly.  A resident latest-only
        stream could hand shot K+1 a late frame from shot K -- a timed-out
        trigger whose frame arrives after the retry fired -- and per-shot
        acquisitions are slow enough that the restart cost does not matter.
        """

        del source_group_sizes, buffer_frame_count
        self.open()
        from pypylon import pylon  # noqa: PLC0415

        camera = self._camera
        if self.config.free_run:
            if not camera.IsGrabbing():
                camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        else:
            if camera.IsGrabbing():
                camera.StopGrabbing()
            if frames is None:
                camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
            else:
                camera.StartGrabbingMax(int(frames), pylon.GrabStrategy_OneByOne)
        self._armed_total = None if frames is None else int(frames)
        self._grabbed = 0
        self._armed = True
        self._capture_incomplete = False
        if timeout:
            self.config = type(self.config)(**{**self.config.__dict__, "timeout_seconds": float(timeout)})

    def read_frame_records(
        self,
        n: int,
        *,
        timeout: float,
        exact: bool,
    ) -> Sequence[CameraFrameRecord]:
        """Retrieve up to ``n`` frames.

        How loudly a missing frame fails follows the trigger mode, and this is
        the single place that decides it.  Free-running, a missing frame is a
        viewer seeing no light: return short and let the live view freeze.
        Hardware-triggered with ``exact``, a missing frame means a trigger was
        lost, which corrupts the shot -- so it raises rather than returning a
        short cycle that downstream would treat as complete.
        """

        if not self._armed:
            raise RuntimeError("camera is not armed")
        camera = self._camera
        records: list[CameraFrameRecord] = []
        deadline = time.monotonic() + float(timeout)

        while len(records) < int(n):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            result = camera.RetrieveResult(
                max(1, int(min(remaining, 0.2) * 1000)),
                _timeout_handling(),
            )
            if result is None:
                continue
            try:
                if not result.GrabSucceeded():
                    self._capture_incomplete = True
                    continue
                image = np.array(result.Array, copy=True)
            finally:
                result.Release()
            self._grabbed += 1
            records.append(CameraFrameRecord(image, self._grabbed - 1))

        if exact and len(records) < int(n):
            raise RuntimeError(
                f"a triggered acquisition expected {n} frames and received "
                f"{len(records)}; a lost trigger corrupts the shot"
            )
        return tuple(records)

    def finish_record_capture(self) -> CameraCaptureTerminalRecord:
        """End the session.  Free-run leaves the stream resident on purpose."""

        camera = self._camera
        if camera is not None and camera.IsGrabbing() and not self.config.free_run:
            camera.StopGrabbing()
        self._armed = False
        return CameraCaptureTerminalRecord(
            self._grabbed,
            True,
            not self._capture_incomplete,
            True,
        )

    def capture_state(self) -> bool:
        return self._armed


def _timeout_handling():
    from pypylon import pylon  # noqa: PLC0415

    return pylon.TimeoutHandling_Return
