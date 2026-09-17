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
from functools import wraps
import threading
import time
from typing import Sequence
from uuid import uuid4

import numpy as np
from zlc_atom.devices import RecordQueue

from ....authoring import AuthoringField, TunableField
from ..roi_grid import snap_roi_axis
from ..contract import (
    CameraAcquisitionMode,
    CameraCaptureTerminalRecord,
    CameraFrameRecord,
    CameraWorkingPoint,
)
from ..photoelectrons import stated_conversion


__all__ = ["PylonCameraAdapter", "PylonCameraConfig"]


def _serialized(method):
    """Keep every public SDK call on this adapter's single command lane."""

    @wraps(method)
    def call(self, *args, **kwargs):
        with self._command_lock:
            return method(self, *args, **kwargs)

    return call


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
        self._command_lock = threading.RLock()
        self._records = RecordQueue("pylon", join_timeout_seconds=config.timeout_seconds)
        self._worker: threading.Thread | None = None
        self._worker_stop = threading.Event()
        self._terminal: CameraCaptureTerminalRecord | None = None
        self._block_id_transport = ""
        self._block_id_wrap: int | None = None
        self._last_block_id: int | None = None
        self._device_session_id = uuid4().hex
        self._settings_epoch = 0
        self._settings_transition_epochs: set[int] = set()
        self._gain_default = float(config.gain_db)
        self._armed = False
        self._configured = False
        self._monitor_mode = False
        self._working_point: CameraWorkingPoint | None = None
        self._requested_settings: dict[str, object] = {}

    # ------------------------------------------------------------------ open

    @_serialized
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
        requested = self.config
        try:
            if self._camera is None:
                self._attach()
            self._apply_pixel_format()
            self._apply_trigger(monitor=False)
            self._apply_exposure(self.config.exposure_seconds)
            self._apply_gain()
            self._apply_roi(self.config.roi_xywh)
            self._configured = True
            self._record_readback()
            self._requested_settings = {
                "exposure_seconds": requested.exposure_seconds,
                "roi_xywh": requested.roi_xywh,
            }
        except BaseException as primary:
            try:
                self.close()
            except BaseException as secondary:
                primary.add_note(f"pylon close after open failure also failed: {secondary}")
            raise

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
        primary: BaseException | None = None
        if self._worker is not None:
            try:
                self.finish_record_capture()
            except BaseException as error:
                primary = error
                if self._worker is not None and self._worker.is_alive():
                    raise
        try:
            with self._command_lock:
                self._close_camera()
        except BaseException as error:
            if primary is None:
                raise
            primary.add_note(f"pylon close also failed: {error}")
        if primary is not None:
            raise primary

    def _close_camera(self) -> None:
        self._working_point = None
        self._requested_settings.clear()
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

                    camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
                return False

        return _Pause()

    def _apply_exposure(self, seconds: float) -> None:
        # Basler exposes ExposureTime in microseconds, and it is legal to change
        # while grabbing -- no stream pause needed.
        self._camera.ExposureTime.SetValue(float(seconds) * 1e6)

    def _apply_gain(self) -> None:
        # Analog gain, like the exposure above it: legal to change while
        # grabbing, so no stream pause.  The camera clamps to its own limits,
        # which is why this asks for the value and then believes the readback.
        self._camera.Gain.SetValue(float(self.config.gain_db))

    def _gain_node(self) -> object:
        """The camera's own gain node, opened if it has to be."""

        self.open()
        return self._camera.Gain

    @_serialized
    def tunable_fields(self) -> tuple[TunableField, ...]:
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
            TunableField(
                metadata=AuthoringField(
                    "gain",
                    "float",
                    "Gain",
                    self._gain_default,
                    minimum=float(node.GetMin()),
                    maximum=float(node.GetMax()),
                    unit="dB",
                ),
                current=float(node.GetValue()),
                live_write=True,
                dependency_group=("gain",),
            ),
        )

    @_serialized
    def tunable_values(self) -> dict[str, float]:
        return {"gain": float(self._gain_node().GetValue())}

    @_serialized
    def settings_provenance(self) -> dict[str, object]:
        return {
            "device_session_id": self._device_session_id,
            "settings_epoch": self._settings_epoch,
        }

    @_serialized
    def tune(self, name: str, value: float) -> float:
        """Move one volunteered knob and return the sensor's effective value."""

        (tunable,) = self.tunable_fields()
        field = tunable.metadata
        if str(name) != field.name:
            raise ValueError(
                f"pylon camera has no tunable field {name!r}; "
                f"it offers {field.name!r}"
            )
        gain = float(value)
        if not np.isfinite(gain) or not (field.minimum <= gain <= field.maximum):
            raise ValueError(
                f"gain must lie in [{field.minimum:g}, {field.maximum:g}] dB"
            )
        node = self._gain_node()
        previous = float(tunable.current)
        if gain != previous:
            self._working_point = None
            node.SetValue(gain)
        effective = float(node.GetValue())
        if effective != self.config.gain_db:
            self._working_point = None
        self.config = replace(self.config, gain_db=effective)
        if effective != previous:
            previous_epoch = self._settings_epoch
            self._settings_epoch += 1
            if self._armed:
                self._settings_transition_epochs.update(
                    (previous_epoch, self._settings_epoch)
                )
        return effective

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

        self._working_point = None
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

    def _apply_roi(self, roi_xywh: tuple[int, int, int, int] | None) -> None:
        """Push the ROI in the GenICam-safe order.

        Zero the offsets, size the window, then place it.  Setting Width while a
        stale OffsetX is still active can violate ``offset + width <= sensor``
        and the camera rejects the write outright -- the same zero-offsets-first
        dance the qCMOS subarray code does.  What the sensor granted is read
        back by ``working_point``, never remembered here.
        """

        camera = self._camera
        with self._paused_stream():
            camera.OffsetX.SetValue(int(camera.OffsetX.GetMin()))
            camera.OffsetY.SetValue(int(camera.OffsetY.GetMin()))
            if roi_xywh is None:
                # Blank means the FULL sensor, never a stale window.
                camera.Width.SetValue(int(camera.WidthMax.GetValue()))
                camera.Height.SetValue(int(camera.HeightMax.GetValue()))
                return
            x, y, width, height = roi_xywh
            sensor_width = int(camera.WidthMax.GetValue())
            sensor_height = int(camera.HeightMax.GetValue())
            # The hardware owns the grid; which way a request is rounded onto
            # it is the same choice for every sensor, and it is made in one
            # place.  This used to round the SIZE down and the offset down
            # too, so the selected region lost its right and bottom edges
            # while its origin moved up and left -- the failure the qCMOS
            # adapter had already been fixed for.
            x, width = snap_roi_axis(
                x,
                width,
                origin_step=int(camera.OffsetX.GetInc()),
                extent_step=int(camera.Width.GetInc()),
                sensor_extent=min(int(camera.Width.GetMax()), sensor_width),
                minimum_extent=int(camera.Width.GetMin()),
            )
            y, height = snap_roi_axis(
                y,
                height,
                origin_step=int(camera.OffsetY.GetInc()),
                extent_step=int(camera.Height.GetInc()),
                sensor_extent=min(int(camera.Height.GetMax()), sensor_height),
                minimum_extent=int(camera.Height.GetMin()),
            )
            camera.Width.SetValue(width)
            camera.Height.SetValue(height)
            camera.OffsetX.SetValue(x)
            camera.OffsetY.SetValue(y)

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

    @_serialized
    def set_exposure_seconds(self, seconds: float) -> CameraWorkingPoint:
        """Integrate for this long on every trigger, leaving the geometry."""

        exposure = float(seconds)
        if not np.isfinite(exposure) or exposure <= 0:
            raise ValueError("exposure_seconds must be positive and finite")
        return self._reconfigure(replace(self.config, exposure_seconds=exposure), "exposure_seconds")

    @_serialized
    def set_roi(
        self, roi_xywh: tuple[int, int, int, int] | None
    ) -> CameraWorkingPoint:
        """Read this part of the sensor, leaving the exposure.

        ``None`` is the whole sensor, which is what an operator means by no
        ROI at all.
        """

        return self._reconfigure(
            replace(self.config, roi_xywh=_roi_request(roi_xywh)), "roi_xywh"
        )

    def _reconfigure(self, candidate: PylonCameraConfig, field: str) -> CameraWorkingPoint:
        """Apply one requested field; keep requested and sensor-quantized facts distinct."""

        if self._armed:
            raise RuntimeError("pylon settings cannot change while armed")
        self.open()
        requested = getattr(candidate, field)
        if field in self._requested_settings and requested == self._requested_settings[field]:
            return self.working_point()
        self._working_point = None
        try:
            if field == "exposure_seconds":
                self._apply_exposure(candidate.exposure_seconds)
            else:
                self._apply_roi(candidate.roi_xywh)
            point = self._record_readback()
        except BaseException as primary:
            self._requested_settings.clear()
            try:
                self._record_readback()
            except BaseException as secondary:
                primary.add_note(
                    f"pylon readback after the refused setting also failed: {secondary}"
                )
            raise
        self._requested_settings[field] = requested
        return point

    def _record_readback(self) -> CameraWorkingPoint:
        """The sensor's own state, as the config.  The whole sensor is ``None``."""

        point = self._read_working_point()
        top, left = point.roi_origin_yx
        height, width = point.roi_shape_yx
        whole_sensor = (top, left) == (0, 0) and (height, width) == point.sensor_shape_yx
        self.config = replace(
            self.config,
            exposure_seconds=point.exposure_seconds,
            roi_xywh=None if whole_sensor else (left, top, width, height),
        )
        self._working_point = point
        return point

    @_serialized
    def working_point(self) -> CameraWorkingPoint:
        """Reuse actual readback until a setting or acquisition mode changes."""
        self.open()
        return self._working_point or self._record_readback()

    def _read_working_point(self) -> CameraWorkingPoint:
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
                "pylon:Mono8;free-running;grab=OneByOne"
                if free_running
                else (
                    f"pylon:Mono8;external={self.config.trigger_source};"
                    "grab=OneByOne"
                )
            ),
            offset_counts=None if conversion is None else conversion[0],
            electrons_per_count=None if conversion is None else conversion[1],
        )

    @_serialized
    def arm(
        self,
        frames: int | None,
        *,
        source_group_sizes: tuple[int, ...] | None,
        buffer_frame_count: int,
        timeout: float,
    ) -> None:
        """Start ordered intake with the requested trigger mode.

        A source-less MONITOR acquisition is free-running only for this arm.
        A repeating source group is the Camera Measurement continuous mode and
        remains externally triggered and ordered.  Finish restores the external
        working point in either case.

        Every mode uses OneByOne and the common bounded FIFO. Finite target
        size never determines how much SDK or application memory is reserved.
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
            sdk_capacity = min(buffer_frame_count, int(camera.MaxNumBuffer.GetMax()))
            camera.MaxNumBuffer.SetValue(sdk_capacity)
            if int(camera.MaxNumBuffer.GetValue()) != sdk_capacity:
                raise RuntimeError("pylon did not apply the requested frame-buffer capacity")
            self._apply_trigger(monitor=monitor)
            self._block_id_transport = str(camera.GetDeviceInfo().GetDeviceClass())
            self._block_id_wrap = None
            if self._block_id_transport == "BaslerGigE":
                nodes = camera.GetNodeMap()
                extended = nodes.GetNode("GevGVSPExtendedIDMode")
                if extended is None:
                    extended = nodes.GetNode("BslGevGVSPExtendedIDMode")
                if extended is None or not bool(extended.GetValue()):
                    self._block_id_wrap = 65535
            self._last_block_id = None
            if expected is None:
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
        self._armed = True
        self._monitor_mode = monitor
        self._working_point = None
        self._records.arm(expected, buffer_record_count=buffer_frame_count)
        self._terminal = None
        self._worker_stop.clear()
        self._worker = threading.Thread(
            target=self._receive, name="zlc-pylon-camera-receiver", daemon=True,
        )
        try:
            self._worker.start()
        except BaseException as error:
            self._worker = None
            try:
                self.finish_record_capture()
            except BaseException as cleanup:
                error.add_note(f"pylon cleanup after receiver startup failure also failed: {cleanup}")
            self._records.fail(error)
            raise

    def read_frame_records(
        self,
        n: int,
        *,
        timeout: float,
        exact: bool,
    ) -> Sequence[CameraFrameRecord]:
        """Consume the accepted FIFO; SDK intake never waits for this call."""
        return tuple(self._records.read(n, timeout=timeout, exact=exact))

    def _receive_one(self, timeout_ms: int) -> None:
        """Copy one SDK result while holding the existing command lock."""
        frame_epochs = tuple(sorted(self._settings_transition_epochs)) or (
            self._settings_epoch,
        )
        result = self._camera.RetrieveResult(timeout_ms, _timeout_handling())
        if result is None or not result.IsValid():
            return
        try:
            if not result.GrabSucceeded():
                raise RuntimeError("a camera acquisition returned a failed frame")
            image = np.asarray(result.Array)
            if image.dtype != np.dtype("uint8"):
                raise RuntimeError(f"pylon Mono8 capture returned dtype {image.dtype}, expected uint8")
            block_id = int(result.GetBlockID())
            previous = self._last_block_id
            if self._block_id_transport in ("BaslerUsb", "BaslerGigE"):
                if block_id == (1 << 64) - 1:
                    raise RuntimeError("pylon returned an invalid hardware BlockID; frame continuity is unprovable")
                # GigE explicitly uses zero for unsupported IDs; USB's first
                # valid ID is zero. Only legacy GigE wraps 65535 back to 1.
                unsupported = self._block_id_transport == "BaslerGigE" and block_id == 0
                if previous is not None:
                    expected_id = 1 if previous == self._block_id_wrap else previous + 1
                    if block_id != expected_id:
                        raise RuntimeError(
                            f"pylon hardware frame sequence gap: expected {expected_id}, received {block_id}; "
                            f"accepted={self._records.produced_count}, capacity={self._records.capacity}, "
                            f"observed_at_ns={time.time_ns()}"
                        )
                if not unsupported:
                    self._last_block_id = block_id
            else:
                # No transport-specific numbering contract is asserted here.
                unsupported = True
            record = CameraFrameRecord(
                image, self._records.produced_count,
                frame_stamp=None if unsupported else block_id,
                settings_session_id=self._device_session_id, settings_epochs=frame_epochs,
            )
        finally:
            result.Release()
        if not self._records.push(record):
            raise self._records.failure or RuntimeError("pylon intake ended before its SDK result")

    def _receive(self) -> None:
        try:
            while not self._worker_stop.is_set():
                with self._command_lock:
                    if self._worker_stop.is_set():
                        break
                    self._receive_one(50)
                    if not self._records.accepting:
                        break
        except BaseException as error:
            self._records.fail(error)
            try:
                with self._command_lock:
                    self._camera.StopGrabbing()
            except BaseException as cleanup:
                error.add_note(f"pylon stop after intake failure also failed: {cleanup}")
        finally:
            self._worker_stop.set()

    def finish_record_capture(self) -> CameraCaptureTerminalRecord:
        """End this arm and restore the finite external-trigger working point."""
        self._worker_stop.set()
        worker = self._worker
        if worker is not None:
            worker.join(timeout=self.timeout)
            if worker.is_alive():
                raise RuntimeError("pylon receive worker did not stop; camera retained")
        with self._command_lock:
            if self._terminal is not None:
                self._records.finish()
                return self._terminal
            camera = self._camera
            if camera is not None:
                # StopGrabbing clears SDK result queues. Stop the sensor first,
                # then accept already-ready results before releasing that queue.
                if self._armed and self._records.failure is None and camera.IsGrabbing():
                    camera.AcquisitionStop.Execute()
                    ready = int(camera.NumReadyBuffers.GetValue())
                    try:
                        for _ in range(ready):
                            self._receive_one(0)
                    except BaseException as error:
                        self._records.fail(error)
                try:
                    self._stop_and_restore_external()
                except BaseException as error:
                    if self._records.failure is not None:
                        self._records.failure.add_note(f"pylon stop also failed: {error}")
                        raise self._records.failure
                    raise
            self._worker = None
            self._armed = False
            self._settings_transition_epochs.clear()
            count = self._records.finish()
            self._terminal = CameraCaptureTerminalRecord(
                count, True, not self._records.pending_count, True,
            )
            return self._terminal

    @_serialized
    def capture_state(self) -> bool:
        return self._armed


def _timeout_handling():
    from pypylon import pylon  # noqa: PLC0415

    return pylon.TimeoutHandling_Return
