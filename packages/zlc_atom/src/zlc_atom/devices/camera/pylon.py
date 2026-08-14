"""Basler pylon camera with finite-trigger and temporary-monitor modes.

Ported from the pre-split tree against this package's ``CameraAdapter``
contract.  Not a byte-copy -- the old driver was written against a different
base class -- but every behaviour that was learned the hard way is carried over,
and each is commented where it lives.

The camera never touches a sequencer.  Triggered finite and continuous arms use
``FrameStart`` from the configured line.  Only a source-less device preview
temporarily uses latest-image free-run, then restores the external working point.

``pypylon`` is imported lazily, so a machine with no Basler runtime still
imports this package, runs the virtual backend, and passes the whole suite.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Sequence

import numpy as np

from ...authoring import AuthoringField
from .contract import (
    CameraAcquisitionMode,
    CameraCaptureTerminalRecord,
    CameraFrameRecord,
    CameraWorkingPoint,
)
from .photoelectrons import stated_conversion


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

    serial: str
    exposure_seconds: float = 0.1
    #: Basler's analog gain, in dB, from the sensor's Analog Control section.
    #: Written down here so an apparatus starts the camera where it was left;
    #: the CAMERA holds the live value from then on, because that is where it
    #: is, and where a scan or the Device Manager moves it.
    gain_db: float = 0.0
    trigger_source: str = "Line1"
    roi_xywh: tuple[int, int, int, int] | None = None
    timeout_seconds: float = 2.0
    #: What one count is worth in photoelectrons, and where zero of them
    #: sits, if this sensor's datasheet says.  Left unset it says nothing,
    #: which is the honest answer for most machine-vision cameras: their
    #: frames are published as the counts they are.
    offset_counts: float | None = None
    electrons_per_count: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.serial, str):
            raise TypeError("pylon serial must be text")
        serial = self.serial.strip()
        if not serial:
            raise ValueError("pylon serial must be non-empty")
        if not isinstance(self.trigger_source, str):
            raise TypeError("pylon trigger_source must be text")
        trigger_source = self.trigger_source.strip()
        if not trigger_source:
            raise ValueError("pylon trigger_source must be non-empty")
        exposure = float(self.exposure_seconds)
        if not np.isfinite(exposure) or exposure <= 0.0:
            raise ValueError("exposure_seconds must be positive and finite")
        gain_db = float(self.gain_db)
        if not np.isfinite(gain_db):
            raise ValueError("gain_db must be finite")
        timeout = float(self.timeout_seconds)
        if not np.isfinite(timeout) or timeout <= 0.0:
            raise ValueError("timeout_seconds must be positive and finite")
        object.__setattr__(self, "serial", serial)
        object.__setattr__(self, "trigger_source", trigger_source)
        object.__setattr__(self, "exposure_seconds", exposure)
        object.__setattr__(self, "gain_db", gain_db)
        object.__setattr__(self, "timeout_seconds", timeout)
        object.__setattr__(self, "roi_xywh", _roi_request(self.roi_xywh))
        stated_conversion(
            self.offset_counts,
            self.electrons_per_count,
            camera="pylon camera",
        )


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
        self._monitor_mode = False

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

        if self._configured:
            return
        try:
            if self._camera is None:
                self._attach()
            self._apply_pixel_format()
            self._apply_trigger(monitor=False)
            self._apply_exposure()
            self._apply_gain()
            self._apply_roi()
        except BaseException as primary:
            try:
                self.close()
            except BaseException as secondary:
                primary.add_note(f"pylon close after open failure also failed: {secondary}")
            raise
        self._configured = True

    def _attach(self) -> None:
        from pypylon import pylon  # noqa: PLC0415 -- lazy on purpose, see open()

        factory = pylon.TlFactory.GetInstance()
        devices = [
            info
            for info in factory.EnumerateDevices()
            if str(info.GetSerialNumber()) == self.config.serial
        ]
        if not devices:
            raise RuntimeError(f"no Basler camera with serial {self.config.serial!r}")
        device = factory.CreateDevice(devices[0])
        camera = pylon.InstantCamera(device)
        try:
            camera.Open()
        except BaseException as primary:
            try:
                camera.Close()
            except BaseException as secondary:
                primary.add_note(f"pylon close after SDK Open failure also failed: {secondary}")
            raise
        self._camera = camera

    def close(self) -> None:
        camera = self._camera
        if camera is None:
            self._armed = False
            self._monitor_mode = False
            self._configured = False
            return
        primary: BaseException | None = None
        try:
            self._stop_and_restore_external()
        except BaseException as error:
            primary = error
        closed = False
        try:
            camera.Close()
            closed = True
        except BaseException as error:
            if primary is None:
                primary = error
            else:
                primary.add_note(f"pylon camera close also failed: {error}")
        if closed:
            self._camera = None
            self._configured = False
            self._armed = False
            self._monitor_mode = False
        if primary is not None:
            raise primary

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

    def _apply_gain(self) -> None:
        # Analog gain, like the exposure above it: legal to change while
        # grabbing, so no stream pause.  The camera clamps to its own limits,
        # which is why this asks for the value and then believes the readback.
        self._camera.Gain.SetValue(float(self.config.gain_db))

    def _gain_node(self) -> object:
        """The camera's own gain node, opened if it has to be."""

        self.open()
        return self._camera.Gain

    def tunable_fields(self) -> tuple[AuthoringField, ...]:
        """The runtime knob this camera volunteers, in the camera's own words.

        Bounds are read from the sensor rather than written down here: they
        differ per model and per pixel format, and a limit this file invents
        is a limit that disagrees with the hardware.  Declared through the
        same AuthoringField every other device uses, so one declaration
        serves a scan axis, the Device Manager form, and nothing else needs
        to know this camera exists.
        """

        node = self._gain_node()
        return (
            AuthoringField(
                "gain_db",
                "float",
                "Gain (dB)",
                float(node.GetValue()),
                minimum=float(node.GetMin()),
                maximum=float(node.GetMax()),
            ),
        )

    def tune(self, name: str, value: float) -> None:
        """Move one volunteered knob on the live camera."""

        (field,) = self.tunable_fields()
        if str(name) != field.name:
            raise ValueError(
                f"pylon camera has no tunable field {name!r}; "
                f"it offers {field.name!r}"
            )
        gain = float(value)
        if not np.isfinite(gain) or not (field.minimum <= gain <= field.maximum):
            raise ValueError(
                f"gain_db must lie in [{field.minimum:g}, {field.maximum:g}]"
            )
        self._camera.Gain.SetValue(gain)

    def _apply_pixel_format(self) -> None:
        with self._paused_stream():
            self._camera.PixelFormat.SetValue("Mono8")
            if str(self._camera.PixelFormat.GetValue()) != "Mono8":
                raise RuntimeError("pylon PixelFormat readback differs from fixed Mono8")

    def _apply_trigger(self, *, monitor: bool) -> None:
        camera = self._camera
        camera.TriggerSelector.SetValue("FrameStart")
        if str(camera.TriggerSelector.GetValue()) != "FrameStart":
            raise RuntimeError("pylon TriggerSelector readback differs from FrameStart")
        if monitor:
            camera.TriggerMode.SetValue("Off")
            if str(camera.TriggerMode.GetValue()) != "Off":
                raise RuntimeError("pylon TriggerMode readback differs from monitor free-run")
        else:
            camera.TriggerMode.SetValue("On")
            camera.TriggerSource.SetValue(str(self.config.trigger_source))
            if str(camera.TriggerMode.GetValue()) != "On":
                raise RuntimeError("pylon TriggerMode readback differs from external trigger")
            if str(camera.TriggerSource.GetValue()) != self.config.trigger_source:
                raise RuntimeError("pylon TriggerSource readback differs from its configured line")
            activation = getattr(camera, "TriggerActivation", None)
            if activation is not None:
                activation.SetValue("RisingEdge")
                if str(activation.GetValue()) != "RisingEdge":
                    raise RuntimeError(
                        "pylon TriggerActivation readback differs from RisingEdge"
                    )

    def _stop_and_restore_external(self) -> None:
        """Attempt both terminal actions and preserve the first failure."""

        camera = self._camera
        if camera is None:
            return
        primary: BaseException | None = None
        try:
            if camera.IsGrabbing():
                camera.StopGrabbing()
            if camera.IsGrabbing():
                raise RuntimeError("pylon remained grabbing after StopGrabbing")
        except BaseException as error:
            primary = error
        try:
            self._apply_trigger(monitor=False)
            self._monitor_mode = False
        except BaseException as error:
            if primary is None:
                primary = error
            else:
                primary.add_note(f"pylon external-trigger restore also failed: {error}")
        if primary is not None:
            raise primary

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

    @property
    def photoelectron_conversion(self) -> tuple[float, float] | None:
        return stated_conversion(
            self.config.offset_counts,
            self.config.electrons_per_count,
            camera="pylon camera",
        )

    def set_exposure_seconds(self, seconds: float) -> CameraWorkingPoint:
        """Integrate for this long on every trigger, leaving the geometry."""

        exposure = float(seconds)
        if not np.isfinite(exposure) or exposure <= 0:
            raise ValueError("exposure_seconds must be positive and finite")
        return self._reconfigure(replace(self.config, exposure_seconds=exposure))

    def set_roi(
        self, roi_xywh: tuple[int, int, int, int] | None
    ) -> CameraWorkingPoint:
        """Read this part of the sensor, leaving the exposure.

        ``None`` is the whole sensor, which is what an operator means by no
        ROI at all.
        """

        return self._reconfigure(
            replace(self.config, roi_xywh=_roi_request(roi_xywh))
        )

    def _reconfigure(self, candidate) -> CameraWorkingPoint:
        """Apply one changed field and keep the config at what the sensor did."""

        if self._armed:
            raise RuntimeError("pylon settings cannot change while armed")
        self.open()
        self.config = candidate
        self._roi = candidate.roi_xywh
        self._apply_exposure()
        self._apply_roi()
        point = self.working_point()
        actual_roi = None
        if candidate.roi_xywh is not None:
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

    def working_point(self) -> CameraWorkingPoint:
        """Read the sensor's state back, rather than repeating what we asked for."""

        self.open()
        camera = self._camera
        width = int(camera.Width.GetValue())
        height = int(camera.Height.GetValue())
        sensor = (int(camera.HeightMax.GetValue()), int(camera.WidthMax.GetValue()))
        origin = (int(camera.OffsetY.GetValue()), int(camera.OffsetX.GetValue()))
        pixel_format = str(camera.PixelFormat.GetValue())
        if pixel_format != "Mono8":
            raise RuntimeError(f"pylon pixel format is {pixel_format!r}, expected 'Mono8'")
        free_running = self._armed and self._monitor_mode
        expected_trigger_mode = "Off" if free_running else "On"
        if str(camera.TriggerMode.GetValue()) != expected_trigger_mode:
            raise RuntimeError("pylon trigger mode changed outside the adapter")
        if expected_trigger_mode == "On" and (
            str(camera.TriggerSource.GetValue()) != self.config.trigger_source
        ):
            raise RuntimeError("pylon trigger source changed outside the adapter")
        exposure = float(camera.ExposureTime.GetValue()) / 1e6
        conversion = self.photoelectron_conversion
        return CameraWorkingPoint(
            acquisition_mode=(
                CameraAcquisitionMode.FREE_RUNNING
                if free_running
                else CameraAcquisitionMode.EXTERNAL_TRIGGERED
            ),
            frame_shape_yx=(height, width),
            sensor_shape_yx=sensor,
            roi_origin_yx=origin,
            roi_shape_yx=(height, width),
            binning_yx=(1, 1),
            dtype=np.dtype("uint8"),
            count_unit="count",
            exposure_seconds=exposure,
            required_external_trigger_interval_seconds=(
                None if free_running else exposure
            ),
            external_trigger_integration_start_offset_seconds=(
                None if free_running else 0.0
            ),
            # Basler states gain in dB; the working point carries the
            # linear factor every reader of it already assumes, so the two
            # cannot be confused (and 1.0 was simply not the camera's answer).
            gain=float(10.0 ** (float(camera.Gain.GetValue()) / 20.0)),
            readout_mode=(
                "pylon:Mono8;free-running;grab=LatestImageOnly"
                if free_running
                else (
                    f"pylon:Mono8;external={self.config.trigger_source};"
                    "grab=OneByOne"
                )
            ),
            offset_counts=None if conversion is None else conversion[0],
            electrons_per_count=None if conversion is None else conversion[1],
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

        A source-less MONITOR acquisition is free-running only for this arm.
        A repeating source group is the Camera Measurement continuous mode and
        remains externally triggered and ordered.  Finish restores the external
        working point in either case.

        HARDWARE TRIGGER gets one bounded session per arm, stopped when done:
        one trigger, one frame, one shot, strictly.  A resident latest-only
        stream could hand shot K+1 a late frame from shot K -- a timed-out
        trigger whose frame arrives after the retry fired -- and per-shot
        acquisitions are slow enough that the restart cost does not matter.
        """

        if frames is None:
            expected = None
            groups = tuple(int(value) for value in (source_group_sizes or ()))
            if groups and (len(groups) != 1 or groups[0] <= 0):
                raise ValueError(
                    "continuous external capture requires one positive source group"
                )
        else:
            if isinstance(frames, bool) or not isinstance(frames, int) or frames <= 0:
                raise ValueError("frames must be a positive integer or None")
            if not isinstance(source_group_sizes, tuple):
                raise TypeError("finite arm requires tuple source_group_sizes")
            if (
                not source_group_sizes
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value <= 0
                    for value in source_group_sizes
                )
                or sum(source_group_sizes) != frames
            ):
                raise ValueError("source_group_sizes must exactly cover frames")
            expected = frames
            groups = source_group_sizes
        if (
            isinstance(buffer_frame_count, bool)
            or not isinstance(buffer_frame_count, int)
            or buffer_frame_count <= 0
        ):
            raise ValueError("buffer_frame_count must be a positive integer")
        if expected is not None and buffer_frame_count != expected:
            raise ValueError(
                "finite buffer_frame_count must equal the complete frame count"
            )
        bounded_timeout = float(timeout)
        if not np.isfinite(bounded_timeout) or bounded_timeout <= 0.0:
            raise ValueError("timeout must be positive and finite")
        self.open()
        from pypylon import pylon  # noqa: PLC0415

        camera = self._camera
        if self._armed:
            raise RuntimeError("pylon camera is already armed")
        monitor = expected is None and not groups
        try:
            if camera.IsGrabbing():
                camera.StopGrabbing()
            if camera.IsGrabbing():
                raise RuntimeError("pylon remained grabbing after StopGrabbing")
            camera.MaxNumBuffer.SetValue(buffer_frame_count)
            if int(camera.MaxNumBuffer.GetValue()) != buffer_frame_count:
                raise RuntimeError("pylon did not apply the requested frame-buffer capacity")
            self._apply_trigger(monitor=monitor)
            if monitor:
                camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            elif expected is None:
                camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
            else:
                camera.StartGrabbingMax(expected, pylon.GrabStrategy_OneByOne)
        except BaseException as primary:
            try:
                self._stop_and_restore_external()
            except BaseException as secondary:
                primary.add_note(
                    f"pylon rollback after arm failure also failed: {secondary}"
                )
            raise
        self._armed_total = expected
        self._grabbed = 0
        self._armed = True
        self._capture_incomplete = False
        self._monitor_mode = monitor

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
            if result is None or not result.IsValid():
                continue
            try:
                if not result.GrabSucceeded():
                    self._capture_incomplete = True
                    if not self._monitor_mode:
                        raise RuntimeError(
                            "a triggered acquisition returned a failed frame"
                        )
                    continue
                image = np.array(result.Array, copy=True)
                if image.dtype != np.dtype("uint8"):
                    raise RuntimeError(
                        f"pylon Mono8 capture returned dtype {image.dtype}, expected uint8"
                    )
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
        """End this arm and restore the finite external-trigger working point."""

        camera = self._camera
        if camera is not None:
            self._stop_and_restore_external()
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
