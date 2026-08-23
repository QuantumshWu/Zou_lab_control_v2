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
* finite acquisition is always external-triggered.  A source-less one-frame
  monitor cycle is temporarily free-running; a multi-frame cycle remains an
  ordered external stream so one shot cannot be assembled from different shots.
* Mono8 is the fixed payload contract, checked from SDK readback rather than
  inferred from whatever operator-authored string happened to be supplied.
"""

from __future__ import annotations

import sys
import threading
import types

import numpy as np
import pytest

from tests.fakes import FakePlane
from zlc_atom.devices.camera import CameraAcquisitionMode, CameraAdapter
from zlc_atom.devices.camera.pylon import PylonCameraAdapter, PylonCameraConfig
from zlc_atom.nodes.camera_measurement import (
    CameraMeasurementNode,
    CameraMeasurementRequest,
)


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


class _FloatNode:
    """A GenICam float node: gain lives on a continuous range, not a grid."""

    def __init__(self, value=0.0, low=0.0, high=24.0):
        self.value = float(value)
        self._low, self._high = float(low), float(high)
        self.writes: list[float] = []

    def SetValue(self, value):
        self.value = float(value)
        self.writes.append(float(value))

    def GetValue(self):
        return self.value

    def GetMin(self):
        return self._low

    def GetMax(self):
        return self._high


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

    def IsValid(self):
        return True

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
        self.Gain = _FloatNode(0.0, 0.0, 24.0)
        self.MaxNumBuffer = _Node(8, 1, 256, 1)
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
    working_point = adapter.working_point()
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
    adapter.set_roi((101, 51, 641, 481))
    point = adapter.set_exposure_seconds(0.012345)
    assert point.exposure_seconds == pytest.approx(0.012345)
    assert point.required_external_trigger_interval_seconds == pytest.approx(0.012345)
    assert point.roi_origin_yx == (50, 100)
    assert point.roi_shape_yx == (480, 640)

    adapter.arm(None, source_group_sizes=None, buffer_frame_count=1, timeout=0.5)
    with pytest.raises(RuntimeError, match="while armed"):
        adapter.set_roi(None)
        adapter.set_exposure_seconds(0.02)
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
    armed_point = adapter.working_point()
    assert armed_point.acquisition_mode is CameraAcquisitionMode.FREE_RUNNING
    assert armed_point.required_external_trigger_interval_seconds is None
    assert armed_point.external_trigger_integration_start_offset_seconds is None
    assert armed_point.readout_mode == "pylon:Mono8;free-running;grab=LatestImageOnly"
    adapter.read_frame_records(1, timeout=0.5, exact=False)
    adapter.finish_record_capture()
    assert not camera.IsGrabbing()
    assert camera.TriggerMode.GetValue() == "On"
    assert camera.TriggerSource.GetValue() == "Line1"


@pytest.mark.parametrize(
    ("frames_per_cycle", "grab_call", "mode", "readout_mode"),
    (
        (
            1,
            "StartGrabbing(latest)",
            CameraAcquisitionMode.FREE_RUNNING,
            "pylon:Mono8;free-running;grab=LatestImageOnly",
        ),
        (
            3,
            "StartGrabbing(one)",
            CameraAcquisitionMode.EXTERNAL_TRIGGERED,
            "pylon:Mono8;external=Line1;grab=OneByOne",
        ),
    ),
)
def test_camera_measurement_monitor_arms_the_pylon_mode_for_a_whole_cycle(
    fake_pypylon,
    frames_per_cycle: int,
    grab_call: str,
    mode: CameraAcquisitionMode,
    readout_mode: str,
) -> None:
    camera = _FakeCamera(
        frames=[np.zeros((16, 16), np.uint8)] * frames_per_cycle
    )
    adapter = PylonCameraAdapter(_config(), camera=camera)
    plane = FakePlane()
    monitor = None
    try:
        node = CameraMeasurementNode(
            camera=adapter,
            request=CameraMeasurementRequest(
                camera_key="camera",
                exposure_seconds=0.01,
                roi_xywh=(0, 0, 16, 16),
                repeat=0,
                frames_per_cycle=frames_per_cycle,
            ),
            signal_plane=plane,
        )
        monitor = node.monitor()

        assert camera.grab_calls[-1] == grab_call
        assert camera.MaxNumBuffer.GetValue() == 4 * frames_per_cycle
        actual = node.actual_working_point
        assert actual is not None
        assert actual.acquisition_mode is mode
        assert actual.readout_mode == readout_mode
        if frames_per_cycle == 1:
            assert actual.required_external_trigger_interval_seconds is None
        else:
            assert actual.required_external_trigger_interval_seconds == pytest.approx(
                0.01
            )

        for _ in range(frames_per_cycle):
            assert monitor.poll() is not None
        value = plane.freeze().value(node.signal_key("frames"))
        assert value is not None
        archived = value.run_record["device_snapshots"]["camera"]
        assert archived["acquisition_mode"] == mode.value
        assert archived["readout_mode"] == readout_mode
    finally:
        if monitor is not None:
            monitor.close()
        plane.close()


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
    assert camera.MaxNumBuffer.writes[-1] == 3
    continuous_point = adapter.working_point()
    assert continuous_point.acquisition_mode is CameraAcquisitionMode.EXTERNAL_TRIGGERED
    assert continuous_point.required_external_trigger_interval_seconds == pytest.approx(
        adapter.config.exposure_seconds
    )
    assert continuous_point.readout_mode == "pylon:Mono8;external=Line1;grab=OneByOne"
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
        adapter.working_point()


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


def test_the_camera_volunteers_its_analog_gain_and_reports_the_real_one() -> None:
    """The one knob a Basler offers here is the one the sensor actually has.

    Its limits are the camera's, read back rather than written down, and the
    working point stops claiming a gain of 1.0 over whatever the sensor was
    set to.
    """

    camera = _FakeCamera()
    adapter = PylonCameraAdapter(
        PylonCameraConfig(serial="1", exposure_seconds=0.01, gain_db=6.0),
        camera=camera,
    )
    adapter.open()
    assert camera.Gain.writes == [6.0], "the authored gain never reached the sensor"

    (tunable,) = adapter.tunable_fields()
    field = tunable.metadata
    assert field.name == "gain_db"
    assert (field.minimum, field.maximum) == (0.0, 24.0)
    assert field.default == pytest.approx(6.0)
    assert tunable.current == pytest.approx(6.0)
    assert tunable.live_write is True
    assert tunable.dependency_group == ("gain_db",)
    provenance = adapter.settings_provenance()
    assert provenance["settings_epoch"] == 0
    assert adapter.tunable_values() == {"gain_db": pytest.approx(6.0)}

    assert adapter.tune("gain_db", 12.0) == pytest.approx(12.0)
    assert camera.Gain.GetValue() == pytest.approx(12.0)
    refreshed = adapter.tunable_fields()[0]
    assert refreshed.metadata.default == pytest.approx(6.0)
    assert refreshed.current == pytest.approx(12.0)
    assert adapter.tunable_values() == {"gain_db": pytest.approx(12.0)}
    assert adapter.settings_provenance() == {
        "device_session_id": provenance["device_session_id"],
        "settings_epoch": 1,
    }
    assert adapter.tune("gain_db", 12.0) == pytest.approx(12.0)
    assert adapter.settings_provenance()["settings_epoch"] == 1
    # A working point carries the LINEAR factor, which is what every reader of
    # one means by gain; the camera states dB.
    assert adapter.working_point().gain == pytest.approx(10.0 ** (12.0 / 20.0))

    with pytest.raises(ValueError, match="gain_db must lie in"):
        adapter.tune("gain_db", 30.0)
    assert adapter.settings_provenance()["settings_epoch"] == 1
    with pytest.raises(ValueError, match="no tunable field"):
        adapter.tune("exposure_seconds", 0.1)
    other = PylonCameraAdapter(_config(), camera=_FakeCamera())
    assert (
        other.settings_provenance()["device_session_id"]
        != provenance["device_session_id"]
    )


def test_live_gain_and_acquisition_readback_share_one_sdk_lane() -> None:
    camera = _FakeCamera()
    adapter = PylonCameraAdapter(_config(), camera=camera)
    adapter.open()
    camera._grabbing = True
    entered = threading.Event()
    release = threading.Event()
    readback_done = threading.Event()
    errors: list[BaseException] = []
    original_set = camera.Gain.SetValue

    def blocked_set(value: float) -> None:
        entered.set()
        if not release.wait(2.0):
            raise TimeoutError("test did not release the gain write")
        original_set(value)

    camera.Gain.SetValue = blocked_set

    def tune() -> None:
        try:
            adapter.tune("gain_db", 8.0)
        except BaseException as error:
            errors.append(error)

    def readback() -> None:
        try:
            adapter.working_point()
        except BaseException as error:
            errors.append(error)
        finally:
            readback_done.set()

    tune_thread = threading.Thread(target=tune)
    readback_thread = threading.Thread(target=readback)
    tune_thread.start()
    assert entered.wait(1.0)
    readback_thread.start()
    assert not readback_done.wait(0.05), "SDK readback bypassed the live-tune lock"
    release.set()
    tune_thread.join(2.0)
    readback_thread.join(2.0)

    assert not tune_thread.is_alive() and not readback_thread.is_alive()
    assert not errors
    assert camera.IsGrabbing() is True
    assert "StopGrabbing" not in camera.grab_calls


def test_live_gain_change_marks_the_first_read_as_a_conservative_transition(
    fake_pypylon,
) -> None:
    camera = _FakeCamera(frames=[np.zeros((4, 4), np.uint8)] * 3)
    adapter = PylonCameraAdapter(_config(), camera=camera)
    adapter.open()
    session_id = adapter.settings_provenance()["device_session_id"]
    adapter.arm(2, source_group_sizes=(2,), buffer_frame_count=2, timeout=0.5)
    adapter.tune("gain_db", 8.0)
    changed = adapter.read_frame_records(2, timeout=0.5, exact=True)
    assert {record.settings_session_id for record in changed} == {session_id}
    assert {record.settings_epochs for record in changed} == {(0, 1)}
    adapter.finish_record_capture()

    adapter.arm(1, source_group_sizes=(1,), buffer_frame_count=1, timeout=0.5)
    stable = adapter.read_frame_records(1, timeout=0.5, exact=True)
    assert stable[0].settings_epochs == (1,)
    adapter.finish_record_capture()
