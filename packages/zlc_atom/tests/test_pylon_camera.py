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
* how loudly a missing frame fails follows the TRIGGER MODE.  Free-running, it
  is a viewer seeing no light: return short and freeze.  Hardware-triggered, it
  is a lost trigger that corrupts the shot: raise.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

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
    def __init__(self, frames=(), succeed=True):
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

    def Open(self):
        self.opened = True

    def Close(self):
        self.opened = False

    def IsGrabbing(self):
        return self._grabbing

    def StartGrabbing(self, _strategy):
        self._grabbing = True
        self.grab_calls.append("StartGrabbing")

    def StartGrabbingMax(self, count, _strategy):
        self._grabbing = True
        self.grab_calls.append(f"StartGrabbingMax({count})")

    def StopGrabbing(self):
        self._grabbing = False
        self.grab_calls.append("StopGrabbing")

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


def test_a_roi_configured_before_open_reaches_the_sensor(fake_pypylon) -> None:
    """Devices open last, so configure-then-open is the normal order."""

    camera = _FakeCamera()
    adapter = PylonCameraAdapter(
        PylonCameraConfig(roi_xywh=(100, 50, 640, 480), exposure_seconds=0.01),
        camera=camera,
    )
    adapter.open()

    working_point = adapter.capture_working_point()
    assert working_point.roi_shape_yx == (480, 640)
    assert working_point.roi_origin_yx == (50, 100)
    assert abs(working_point.exposure_seconds - 0.01) < 1e-9


def test_the_roi_is_written_offsets_first_and_snapped_to_the_camera_grid(fake_pypylon) -> None:
    camera = _FakeCamera()
    adapter = PylonCameraAdapter(
        # 641 is not a multiple of the width increment 4; 51 not of the y increment 2.
        PylonCameraConfig(roi_xywh=(101, 51, 641, 481)),
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
    PylonCameraAdapter(PylonCameraConfig(roi_xywh=None), camera=camera).open()
    assert camera.Width.GetValue() == camera.WidthMax.GetValue()
    assert camera.Height.GetValue() == camera.HeightMax.GetValue()


def test_free_run_keeps_the_stream_resident_across_arm_and_finish(fake_pypylon) -> None:
    """Restarting a USB3 stream per frame was the live-monitor stutter."""

    camera = _FakeCamera(frames=[np.zeros((4, 4), np.uint8)] * 4)
    adapter = PylonCameraAdapter(PylonCameraConfig(trigger_source="Software"), camera=camera)
    adapter.open()

    adapter.arm(None, source_group_sizes=None, buffer_frame_count=1, timeout=0.5)
    adapter.read_frame_records(1, timeout=0.5, exact=False)
    adapter.finish_record_capture()
    assert camera.IsGrabbing(), "free-run leaves the stream up on purpose"

    adapter.arm(None, source_group_sizes=None, buffer_frame_count=1, timeout=0.5)
    assert camera.grab_calls.count("StartGrabbing") == 1, "the stream restarted"


def test_a_triggered_session_is_bounded_and_stops_when_done(fake_pypylon) -> None:
    """One trigger, one frame, one shot -- strictly.

    A resident latest-only stream could hand shot K+1 a late frame from shot K.
    """

    camera = _FakeCamera(frames=[np.zeros((4, 4), np.uint8)] * 3)
    adapter = PylonCameraAdapter(PylonCameraConfig(trigger_source="Line1"), camera=camera)
    adapter.open()

    adapter.arm(3, source_group_sizes=(3,), buffer_frame_count=3, timeout=0.5)
    assert "StartGrabbingMax(3)" in camera.grab_calls
    adapter.read_frame_records(3, timeout=0.5, exact=True)
    adapter.finish_record_capture()
    assert not camera.IsGrabbing()


def test_fault_loudness_follows_the_trigger_mode(fake_pypylon) -> None:
    """Free-run returns short; a lost hardware trigger raises."""

    quiet = PylonCameraAdapter(PylonCameraConfig(trigger_source="Software"), camera=_FakeCamera(frames=[]))
    quiet.open()
    quiet.arm(None, source_group_sizes=None, buffer_frame_count=1, timeout=0.05)
    assert quiet.read_frame_records(2, timeout=0.05, exact=False) == ()

    loud = PylonCameraAdapter(PylonCameraConfig(trigger_source="Line1"), camera=_FakeCamera(frames=[]))
    loud.open()
    loud.arm(2, source_group_sizes=(2,), buffer_frame_count=2, timeout=0.05)
    with pytest.raises(RuntimeError, match="lost trigger"):
        loud.read_frame_records(2, timeout=0.05, exact=True)


def test_the_module_imports_without_a_basler_runtime() -> None:
    """A machine with no pypylon still imports the package and runs everything else."""

    import importlib

    module = importlib.import_module("zlc_atom.devices.camera.pylon")
    assert module.PylonCameraAdapter is PylonCameraAdapter
    assert "pypylon" not in {name.split(".")[0] for name in dir(module)}
