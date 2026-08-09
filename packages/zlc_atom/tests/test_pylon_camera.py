"""Basler pylon adapter contracts, with no Basler runtime present.

A fake ``pypylon`` stands in for the SDK, so the code under test is exactly what
a real run executes and only the hardware layer is substituted.  Three findings
from the pre-split tree are pinned here because a fresh implementation would not
rediscover them:

* configuring BEFORE open must still reach the sensor.  Devices are opened last,
  so configure-then-open is the normal order; a driver that only applies
  settings it receives while open images the full frame while reporting the ROI
  it was given.
* the ROI goes in offsets-first, and every value is snapped to the CAMERA'S own
  min/max/increment.  Writing a width while a stale offset is live violates
  ``offset + width <= sensor`` and the camera rejects it outright.
* finite acquisition is always external-triggered while a monitor arm is a
  temporary free-running session.  One authored trigger setting must not turn
  the whole camera into one mode or the other.
* Mono8 is the fixed payload contract, checked from SDK readback rather than
  inferred from whatever operator-authored string happened to be supplied.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from zlc_atom.devices.camera import CameraAdapter
from zlc_atom.devices.camera.pylon import PylonCameraAdapter, PylonCameraConfig


class _Node:
    """A GenICam node: a value plus the min/max/increment grid the snap consults."""

    def __init__(self, value=0, low=0, high=10_000, increment=1):
        self.value = int(value)
        self._low, self._high, self._increment = int(low), int(high), int(increment)
        self.writes: list[int] = []

    def SetValue(self, value):
        self.value = int(value)
        self.writes.append(int(value))

    def GetValue(self):
        return self.value

    def GetMin(self):
        return self._low

    def GetMax(self):
        return self._high

    def GetInc(self):
        return self._increment


class _StringNode:
    def __init__(self, value=""):
        self.value = value
        self.writes: list[str] = []

    def SetValue(self, value):
        self.value = str(value)
        self.writes.append(str(value))

    def GetValue(self):
        return self.value


class _Result:
    def __init__(self, array, succeeded=True):
        self.Array = array
        self._succeeded = succeeded
        self.released = False

    def GrabSucceeded(self):
        return self._succeeded

    def Release(self):
        self.released = True


class _FakeCamera:
    def __init__(
        self,
        frames=(),
        succeed=True,
        fail_stop_once=False,
        fail_start_once=False,
        ignore_stop_once=False,
        fail_open=False,
    ):
        self.Width = _Node(64, 16, 1920, 4)
        self.Height = _Node(48, 16, 1200, 2)
        self.WidthMax = _Node(1920)
        self.HeightMax = _Node(1200)
        self.OffsetX = _Node(0, 0, 1904, 4)
        self.OffsetY = _Node(0, 0, 1152, 2)
        self.ExposureTime = _Node(1000, 1, 10_000_000, 1)
        self.PixelFormat = _StringNode("Mono8")
        self.TriggerSelector = _StringNode()
        self.TriggerMode = _StringNode()
        self.TriggerSource = _StringNode()
        self._frames = list(frames)
        self._succeed = succeed
        self._grabbing = False
        self.opened = False
        self.grab_calls: list[str] = []
        self.close_calls = 0
        self._fail_stop_once = fail_stop_once
        self._fail_start_once = fail_start_once
        self._ignore_stop_once = ignore_stop_once
        self._fail_open = fail_open

    def Open(self):
        self.opened = True
        if self._fail_open:
            raise RuntimeError("injected SDK open failure")

    def Close(self):
        self.opened = False
        self.close_calls += 1

    def IsGrabbing(self):
        return self._grabbing

    def StartGrabbing(self, strategy):
        if self._fail_start_once:
            self._fail_start_once = False
            raise RuntimeError("injected start failure")
        self._grabbing = True
        self.grab_calls.append(f"StartGrabbing({strategy})")

    def StartGrabbingMax(self, count, _strategy):
        self._grabbing = True
        self.grab_calls.append(f"StartGrabbingMax({count})")

    def StopGrabbing(self):
        self.grab_calls.append("StopGrabbing")
        if self._ignore_stop_once:
            self._ignore_stop_once = False
            return
        self._grabbing = False
        if self._fail_stop_once:
            self._fail_stop_once = False
            raise RuntimeError("injected stop failure")

    def RetrieveResult(self, _timeout_ms, _handling):
        if not self._frames:
            return None
        return _Result(self._frames.pop(0), succeeded=self._succeed)


@pytest.fixture
def fake_pypylon(monkeypatch):
    pylon = types.SimpleNamespace(
        GrabStrategy_LatestImageOnly="latest",
        GrabStrategy_OneByOne="one",
        TimeoutHandling_Return="return",
    )
    module = types.ModuleType("pypylon")
    module.pylon = pylon
    monkeypatch.setitem(sys.modules, "pypylon", module)
    monkeypatch.setitem(sys.modules, "pypylon.pylon", pylon)
    return pylon


def _config(**values) -> PylonCameraConfig:
    return PylonCameraConfig(serial="PYLON-001", **values)


def test_serial_is_required_and_the_payload_contract_is_not_authorable() -> None:
    with pytest.raises(ValueError, match="serial"):
        PylonCameraConfig(serial="")
    with pytest.raises(TypeError, match="pixel_format"):
        PylonCameraConfig(serial="PYLON-001", pixel_format="Mono12")
    assert _config().trigger_source == "Line1"


def test_a_roi_configured_before_open_reaches_the_sensor(fake_pypylon) -> None:
    """Devices open last, so configure-then-open is the normal order."""

    camera = _FakeCamera()
    adapter = PylonCameraAdapter(
        _config(roi_xywh=(100, 50, 640, 480), exposure_seconds=0.01),
        camera=camera,
    )
    adapter.open()

    assert isinstance(adapter, CameraAdapter)
    working_point = adapter.capture_working_point()
    assert working_point.roi_shape_yx == (480, 640)
    assert working_point.roi_origin_yx == (50, 100)
    assert abs(working_point.exposure_seconds - 0.01) < 1e-9


def test_the_roi_is_written_offsets_first_and_snapped_to_the_camera_grid(fake_pypylon) -> None:
    camera = _FakeCamera()
    adapter = PylonCameraAdapter(
        # 641 is not a multiple of the width increment 4; 51 not of the y increment 2.
        _config(roi_xywh=(101, 51, 641, 481)),
        camera=camera,
    )
    adapter.open()

    # Offsets are zeroed before the window is sized, or the camera rejects the write.
    assert camera.OffsetX.writes[0] == camera.OffsetX.GetMin()
    assert camera.OffsetY.writes[0] == camera.OffsetY.GetMin()
    # Every value lands on the camera's own grid, not a value we invented.
    assert camera.Width.GetValue() % camera.Width.GetInc() == 0
    assert camera.Height.GetValue() % camera.Height.GetInc() == 0
    assert camera.OffsetX.GetValue() % camera.OffsetX.GetInc() == 0


def test_a_blank_roi_means_the_full_sensor_not_a_stale_window(fake_pypylon) -> None:
    camera = _FakeCamera()
    PylonCameraAdapter(_config(roi_xywh=None), camera=camera).open()
    assert camera.Width.GetValue() == camera.WidthMax.GetValue()
    assert camera.Height.GetValue() == camera.HeightMax.GetValue()


def test_measurement_configuration_returns_sdk_readback_and_is_idle_only(fake_pypylon) -> None:
    camera = _FakeCamera()
    adapter = PylonCameraAdapter(
        _config(),
        camera=camera,
    )
    adapter.open()
    point = adapter.configure_measurement(
        exposure_seconds=0.012345,
        roi_xywh=(101, 51, 641, 481),
    )
    assert point.exposure_seconds == pytest.approx(0.012345)
    assert point.required_external_trigger_interval_seconds == pytest.approx(0.012345)
    assert point.roi_origin_yx == (50, 100)
    assert point.roi_shape_yx == (480, 640)

    adapter.arm(None, source_group_sizes=None, buffer_frame_count=1, timeout=0.5)
    with pytest.raises(RuntimeError, match="while armed"):
        adapter.configure_measurement(exposure_seconds=0.02, roi_xywh=None)
    adapter.finish_record_capture()


def test_monitor_arm_is_temporarily_free_running_then_restores_external_trigger(fake_pypylon) -> None:
    """Monitor mode is an arm policy, not the camera's permanent configuration."""

    camera = _FakeCamera(frames=[np.zeros((4, 4), np.uint8)] * 4)
    adapter = PylonCameraAdapter(_config(), camera=camera)
    adapter.open()
    assert camera.TriggerMode.GetValue() == "On"
    assert camera.TriggerSource.GetValue() == "Line1"

    adapter.arm(None, source_group_sizes=None, buffer_frame_count=1, timeout=0.5)
    assert camera.TriggerMode.GetValue() == "Off"
    adapter.read_frame_records(1, timeout=0.5, exact=False)
    adapter.finish_record_capture()
    assert not camera.IsGrabbing()
    assert camera.TriggerMode.GetValue() == "On"
    assert camera.TriggerSource.GetValue() == "Line1"


def test_triggered_finite_and_repeat_zero_sessions_both_preserve_frame_order(
    fake_pypylon,
) -> None:
    """One trigger, one frame, one shot -- strictly.

    A resident latest-only stream could hand shot K+1 a late frame from shot K.
    """

    camera = _FakeCamera(frames=[np.zeros((4, 4), np.uint8)] * 6)
    adapter = PylonCameraAdapter(_config(), camera=camera)
    adapter.open()

    adapter.arm(3, source_group_sizes=(3,), buffer_frame_count=3, timeout=0.5)
    assert camera.TriggerMode.GetValue() == "On"
    assert camera.TriggerSource.GetValue() == "Line1"
    assert "StartGrabbingMax(3)" in camera.grab_calls
    adapter.read_frame_records(3, timeout=0.5, exact=True)
    adapter.finish_record_capture()
    assert not camera.IsGrabbing()

    adapter.arm(None, source_group_sizes=(3,), buffer_frame_count=3, timeout=0.5)
    assert camera.TriggerMode.GetValue() == "On"
    assert camera.TriggerSource.GetValue() == "Line1"
    assert camera.grab_calls[-1] == "StartGrabbing(one)"
    assert len(adapter.read_frame_records(3, timeout=0.5, exact=False)) == 3
    adapter.finish_record_capture()
    assert not camera.IsGrabbing()


def test_fault_loudness_follows_the_arm_mode(fake_pypylon) -> None:
    """A free-run preview returns short; every triggered mode fails loud."""

    quiet = PylonCameraAdapter(_config(), camera=_FakeCamera(frames=[]))
    quiet.open()
    quiet.arm(None, source_group_sizes=None, buffer_frame_count=1, timeout=0.05)
    assert quiet.read_frame_records(2, timeout=0.05, exact=False) == ()

    loud = PylonCameraAdapter(_config(), camera=_FakeCamera(frames=[]))
    loud.open()
    loud.arm(2, source_group_sizes=(2,), buffer_frame_count=2, timeout=0.05)
    with pytest.raises(RuntimeError, match="lost trigger"):
        loud.read_frame_records(2, timeout=0.05, exact=True)

    continuous = PylonCameraAdapter(
        _config(),
        camera=_FakeCamera(frames=[np.zeros((4, 4), np.uint8)], succeed=False),
    )
    continuous.open()
    continuous.arm(
        None,
        source_group_sizes=(2,),
        buffer_frame_count=2,
        timeout=0.05,
    )
    with pytest.raises(RuntimeError, match="failed frame"):
        continuous.read_frame_records(2, timeout=0.05, exact=False)
    quiet.finish_record_capture()
    loud.finish_record_capture()
    continuous.finish_record_capture()


def test_mono8_is_verified_from_sdk_readback(fake_pypylon) -> None:
    camera = _FakeCamera()
    adapter = PylonCameraAdapter(_config(), camera=camera)
    adapter.open()
    camera.PixelFormat.value = "Mono12"
    with pytest.raises(RuntimeError, match="Mono8"):
        adapter.capture_working_point()


def test_open_failure_closes_once_and_close_remains_idempotent(fake_pypylon) -> None:
    camera = _FakeCamera()
    camera.PixelFormat.SetValue = lambda _value: None
    camera.PixelFormat.value = "Mono12"
    adapter = PylonCameraAdapter(_config(), camera=camera)
    with pytest.raises(RuntimeError, match="Mono8"):
        adapter.open()
    assert camera.close_calls == 1
    adapter.close()
    assert camera.close_calls == 1


def test_sdk_open_failure_closes_the_unadopted_candidate(fake_pypylon) -> None:
    candidate = _FakeCamera(fail_open=True)
    info = types.SimpleNamespace(GetSerialNumber=lambda: "PYLON-001")
    factory = types.SimpleNamespace(
        EnumerateDevices=lambda: (info,),
        CreateDevice=lambda selected: selected,
    )
    fake_pypylon.TlFactory = types.SimpleNamespace(GetInstance=lambda: factory)
    fake_pypylon.InstantCamera = lambda _device: candidate

    adapter = PylonCameraAdapter(_config())
    with pytest.raises(RuntimeError, match="SDK open failure"):
        adapter.open()

    assert candidate.close_calls == 1
    adapter.close()
    assert candidate.close_calls == 1


def test_finish_attempts_external_restore_even_if_stream_stop_fails(fake_pypylon) -> None:
    camera = _FakeCamera(fail_stop_once=True)
    adapter = PylonCameraAdapter(_config(), camera=camera)
    adapter.open()
    adapter.arm(None, source_group_sizes=None, buffer_frame_count=1, timeout=0.5)
    with pytest.raises(RuntimeError, match="stop failure"):
        adapter.finish_record_capture()
    assert camera.TriggerMode.GetValue() == "On"
    assert camera.TriggerSource.GetValue() == "Line1"
    adapter.finish_record_capture()
    adapter.close()
    adapter.close()
    assert camera.close_calls == 1


def test_failed_monitor_start_restores_the_external_working_point(fake_pypylon) -> None:
    camera = _FakeCamera(fail_start_once=True)
    adapter = PylonCameraAdapter(_config(), camera=camera)
    adapter.open()

    with pytest.raises(RuntimeError, match="start failure"):
        adapter.arm(None, source_group_sizes=None, buffer_frame_count=1, timeout=0.5)

    assert camera.TriggerMode.GetValue() == "On"
    assert camera.TriggerSource.GetValue() == "Line1"
    adapter.close()


def test_partial_monitor_trigger_write_is_rolled_back(fake_pypylon) -> None:
    camera = _FakeCamera()
    adapter = PylonCameraAdapter(_config(), camera=camera)
    adapter.open()
    original_get = camera.TriggerMode.GetValue
    failed = False

    def fail_first_monitor_readback():
        nonlocal failed
        if camera.TriggerMode.value == "Off" and not failed:
            failed = True
            raise RuntimeError("injected trigger readback failure")
        return original_get()

    camera.TriggerMode.GetValue = fail_first_monitor_readback
    with pytest.raises(RuntimeError, match="trigger readback failure"):
        adapter.arm(None, source_group_sizes=None, buffer_frame_count=1, timeout=0.5)

    assert camera.TriggerMode.GetValue() == "On"
    assert camera.TriggerSource.GetValue() == "Line1"
    adapter.close()


def test_finish_refuses_to_claim_terminal_while_the_sdk_is_still_grabbing(
    fake_pypylon,
) -> None:
    camera = _FakeCamera(ignore_stop_once=True)
    adapter = PylonCameraAdapter(_config(), camera=camera)
    adapter.open()
    adapter.arm(None, source_group_sizes=None, buffer_frame_count=1, timeout=0.5)

    with pytest.raises(RuntimeError, match="remained grabbing"):
        adapter.finish_record_capture()

    assert adapter.capture_state() is True
    assert camera.IsGrabbing() is True
    adapter.finish_record_capture()
    assert adapter.capture_state() is False


def test_the_module_imports_without_a_basler_runtime() -> None:
    """A machine with no pypylon still imports the package and runs everything else."""

    import importlib

    module = importlib.import_module("zlc_atom.devices.camera.pylon")
    assert module.PylonCameraAdapter is PylonCameraAdapter
    assert "pypylon" not in {name.split(".")[0] for name in dir(module)}
