from __future__ import annotations

import ctypes
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from zlc_atom.devices.slm.device_types import (
    DEVICE_TYPES,
    HAMAMATSU_X15213_SCHEMA,
    X15213Adapter,
    _DevModeW,
    _DisplayDeviceW,
    _discover_x15213,
)
from zlc_atom.install import create_installation


class _UsbSdk:
    def __init__(
        self,
        *,
        mode: int = 1,
        serial_result: int = 1,
        mode_result: int = 1,
        close_results: tuple[int, ...] = (),
    ) -> None:
        self.mode = mode
        self.serial = b"LSH0804382"
        self.serial_result = serial_result
        self.mode_result = mode_result
        self.display = np.zeros((1024, 1272), dtype=np.uint8)
        self.closed = False
        self.close_count = 0
        self.close_results = list(close_results)
        self.selected = None
        self.rebooted = False
        self.bad_readback = False

    def Open_Dev(self, ids, size):
        ids[0] = 7
        return 1

    def Check_HeadSerial(self, _board, target, _size):
        if self.serial_result != 1:
            return self.serial_result
        target.value = self.serial
        return 1

    def Mode_Check(self, _board, target):
        if self.mode_result != 1:
            return self.mode_result
        target._obj.value = self.mode
        return 1

    def Mode_Select(self, _board, mode):
        self.selected = int(mode)
        self.mode = int(mode)
        return 1

    def Reboot(self, _board):
        self.rebooted = True
        return 1

    def Write_FMemArray(self, _board, source, size, width, height, _slot):
        assert (int(height), int(width)) == self.display.shape
        self.display = np.ctypeslib.as_array(source, shape=(int(size),)).reshape(
            self.display.shape
        ).copy()
        return 1

    def Change_DispSlot(self, _board, _slot):
        return 1

    def Check_Disp_IMG(self, _board, size, width, height, target):
        result = self.display.copy()
        if self.bad_readback:
            result[0, 0] ^= np.uint8(1)
        np.ctypeslib.as_array(target, shape=(int(size),))[:] = result.reshape(-1)
        return 1

    def Close_Dev(self, _ids, _size):
        self.close_count += 1
        result = self.close_results.pop(0) if self.close_results else 1
        if result == 1:
            self.closed = True
        return result


class _Handle:
    def __init__(self, *, failures: int = 0) -> None:
        self.close_count = 0
        self.failures = failures

    def close(self) -> None:
        self.close_count += 1
        if self.failures:
            self.failures -= 1
            raise RuntimeError("DLL directory handle close failed")


def _display_endpoint(name: str = r"\\.\DISPLAY2") -> dict[str, object]:
    return {
        "name": name,
        "attached": True,
        "width": 1280,
        "height": 1024,
        "frequency": 60,
        "x": 1920,
        "y": 0,
    }


def _patch_dvi_without_controller(monkeypatch) -> None:
    import zlc_atom.devices.slm.device_types as module

    monkeypatch.setattr(module, "_windows_displays", lambda: (_display_endpoint(),))
    monkeypatch.setattr(module, "_find_sdk_directory", lambda _authored="": None)


def _config(**changes):
    values = {
        "transport": "dvi",
        "display_name": r"\\.\DISPLAY2",
        "sdk_directory": "",
        "serial": "",
        "device_profile": "LSH0804382",
        "wavelength_nm": 852.0,
        "correction_path": "",
        "flip_x": False,
        "flip_y": False,
        "settle_seconds": 0.0,
    }
    values.update(changes)
    return values


def test_x15213_descriptor_is_one_plugin_owned_usb_or_dvi_device() -> None:
    assert len(DEVICE_TYPES) == 1
    descriptor = DEVICE_TYPES[0]
    assert descriptor.type_id == "slm.hamamatsu_x15213"
    assert descriptor.domain == "slm"
    assert descriptor.capabilities == ("slm.phase",)
    assert descriptor.discover is _discover_x15213
    assert descriptor.control_factory is not None
    assert set(HAMAMATSU_X15213_SCHEMA.field_names) == set(_config())
    assert {
        "model", "two_pi_gray", "correction_offset_gray", "correction_sign",
        "active_x",
    }.isdisjoint(HAMAMATSU_X15213_SCHEMA.field_names)
    with pytest.raises(ValueError, match="transport"):
        HAMAMATSU_X15213_SCHEMA.project_values(_config(transport="serial"))


def test_x15213_profile_derives_wavelength_lut_without_a_hard_coded_level(
    monkeypatch,
) -> None:
    import zlc_atom.devices.slm.device_types as module

    _patch_dvi_without_controller(monkeypatch)
    adapter = X15213Adapter(_config())
    try:
        assert adapter.wavelength_nm == 852.0
        assert adapter.two_pi_gray == 225.0
    finally:
        adapter.close()

    at_readout = X15213Adapter(_config(wavelength_nm=785.0))
    try:
        assert at_readout.two_pi_gray == 207.0
    finally:
        at_readout.close()

    profile_path = Path(module.__file__).resolve().parent / "profiles" / "LSH0804382.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert len(profile["phase_pi_by_gray"]) == 256
    assert "wavelength_levels" not in profile
    assert "two_pi_gray" not in profile

    with pytest.raises(ValueError, match="wavelength.*phase-curve range"):
        X15213Adapter(_config(wavelength_nm=1000.0))
    with pytest.raises((FileNotFoundError, ValueError), match="profile"):
        X15213Adapter(_config(device_profile="not-a-device"))
    with pytest.raises(ValueError, match="serial.*profile"):
        X15213Adapter(_config(serial="ANOTHER-HEAD"))


def test_win32_display_structures_match_the_complete_unicode_abi() -> None:
    assert ctypes.sizeof(_DisplayDeviceW) == 840
    assert ctypes.sizeof(_DevModeW) == 220
    assert _DevModeW.dmPositionX.offset == 76
    assert _DevModeW.dmPelsWidth.offset == 172
    assert _DevModeW.dmPanningHeight.offset == 216


def test_x15213_factory_binds_without_private_test_configuration(
    monkeypatch,
) -> None:
    import zlc_atom.devices.slm.device_types as module

    _patch_dvi_without_controller(monkeypatch)
    frames: list[np.ndarray] = []
    closed: list[bool] = []
    monkeypatch.setattr(
        module,
        "_open_dvi_presenter",
        lambda _name: (
            lambda frame: frames.append(frame.copy()),
            lambda: closed.append(True),
        ),
    )
    installation = create_installation(
        (
            {
                "key": "slm",
                "type_id": "slm.hamamatsu_x15213",
                "config": _config(),
            },
        )
    )
    try:
        assert installation.failures == {}
        slm = installation.capability("slm.phase", key="slm")
        slm.apply_phase(np.zeros(slm.shape_yx))
        assert frames[-1].shape == (1024, 1280)
    finally:
        installation.close()
    assert closed == [True]
    assert {"presenter", "sdk", "sleep"}.isdisjoint(
        inspect.signature(X15213Adapter).parameters
    )


def test_x15213_factory_rejects_private_presenter_configuration() -> None:
    installation = create_installation(
        (
            {
                "key": "slm",
                "type_id": "slm.hamamatsu_x15213",
                "config": {**_config(), "_presenter": object()},
            },
        )
    )
    try:
        assert "slm" in installation.failures
        assert "unknown X15213 configuration fields" in str(installation.failures["slm"])
    finally:
        installation.close()


def test_scan_reports_dvi_candidates_without_claiming_edid_identity(monkeypatch) -> None:
    import zlc_atom.devices.slm.device_types as module

    monkeypatch.setattr(
        module,
        "_windows_displays",
        lambda: (
            {
                "name": r"\\.\DISPLAY1",
                "attached": True,
                "width": 1920,
                "height": 1080,
                "frequency": 60,
                "x": 0,
                "y": 0,
            },
            {
                "name": r"\\.\DISPLAY2",
                "attached": True,
                "width": 1280,
                "height": 1024,
                "frequency": 60,
                "x": 1920,
                "y": 0,
            },
            {
                "name": r"\\.\DISPLAY3",
                "attached": False,
                "width": 1280,
                "height": 1024,
                "frequency": 60,
                "x": -1280,
                "y": 0,
            },
        ),
    )
    monkeypatch.setattr(module, "_find_sdk_directory", lambda _authored="": None)

    found = _discover_x15213()

    assert len(found) == 1
    candidate = found[0]
    assert candidate.type_id == "slm.hamamatsu_x15213"
    assert candidate.parameters["transport"] == "dvi"
    assert candidate.parameters["display_name"] == r"\\.\DISPLAY2"
    assert candidate.parameters["device_profile"] == "LSH0804382"
    assert candidate.parameters["wavelength_nm"] == 852.0
    assert "candidate" in candidate.role


def test_scan_reads_usb_serial_and_always_closes_the_probe(monkeypatch, tmp_path: Path) -> None:
    import zlc_atom.devices.slm.device_types as module

    sdk = _UsbSdk()
    handle = _Handle()

    monkeypatch.setattr(module, "_windows_displays", lambda: ())
    monkeypatch.setattr(module, "_find_sdk_directory", lambda _authored="": tmp_path)
    monkeypatch.setattr(module, "_load_sdk", lambda _directory: (sdk, handle))

    found = _discover_x15213()

    assert len(found) == 1
    assert found[0].parameters["transport"] == "usb"
    assert found[0].parameters["serial"] == "LSH0804382"
    assert found[0].parameters["device_profile"] == "LSH0804382"
    assert found[0].parameters["wavelength_nm"] == 852.0
    assert sdk.closed
    assert sdk.close_count == 1
    assert handle.close_count == 1


def test_scan_keeps_dvi_candidate_when_sdk_has_no_usb_device(monkeypatch, tmp_path: Path) -> None:
    import zlc_atom.devices.slm.device_types as module

    class _NoUsb:
        def Open_Dev(self, _ids, _size):
            return 0

    monkeypatch.setattr(
        module,
        "_windows_displays",
        lambda: (
            {
                "name": r"\\.\DISPLAY2",
                "attached": True,
                "width": 1280,
                "height": 1024,
                "frequency": 60,
                "x": 1920,
                "y": 0,
            },
        ),
    )
    monkeypatch.setattr(module, "_find_sdk_directory", lambda _authored="": tmp_path)
    monkeypatch.setattr(module, "_load_sdk", lambda _directory: (_NoUsb(), None))

    found = _discover_x15213()

    assert len(found) == 1
    assert found[0].parameters["transport"] == "dvi"


def test_runtime_correction_status_and_failed_load_are_atomic(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import zlc_atom.devices.slm.device_types as module

    _patch_dvi_without_controller(monkeypatch)
    frames: list[np.ndarray] = []
    monkeypatch.setattr(
        module,
        "_open_dvi_presenter",
        lambda _name: (lambda frame: frames.append(frame.copy()), lambda: None),
    )
    adapter = X15213Adapter(_config())
    initial_phase = adapter.last_commanded_phase.copy()
    correction_path = tmp_path / "runtime-correction.bmp"
    Image.fromarray(
        np.full((1024, 1272), 7, dtype=np.uint8), mode="L"
    ).save(correction_path)

    try:
        assert adapter.correction_path == ""
        assert adapter.correction_name == ""
        assert adapter.correction_enabled is False
        assert adapter.correction_available is True
        with pytest.raises(RuntimeError, match="no .* correction map"):
            adapter.set_correction_enabled(True)

        adapter.load_correction(correction_path)

        assert adapter.correction_path == str(correction_path)
        assert adapter.correction_name == correction_path.name
        assert adapter.correction_enabled is True
        assert adapter.correction_available is True
        np.testing.assert_array_equal(adapter.last_commanded_phase, initial_phase)
        assert adapter._presenter is None
        assert frames == []

        bad_path = tmp_path / "wrong-shape.bmp"
        Image.fromarray(np.zeros((12, 13), dtype=np.uint8), mode="L").save(bad_path)
        with pytest.raises(ValueError, match="1272 x 1024"):
            adapter.load_correction(bad_path)
        full_raster = tmp_path / "full-raster.bmp"
        Image.fromarray(np.zeros((1024, 1280), dtype=np.uint8), mode="L").save(
            full_raster
        )
        with pytest.raises(ValueError, match="1272 x 1024"):
            adapter.load_correction(full_raster)
        rgb = tmp_path / "rgb.bmp"
        Image.fromarray(np.zeros((1024, 1272, 3), dtype=np.uint8), mode="RGB").save(
            rgb
        )
        with pytest.raises(ValueError, match="8-bit grayscale"):
            adapter.load_correction(rgb)

        assert adapter.correction_path == str(correction_path)
        assert adapter.correction_name == correction_path.name
        assert adapter.correction_enabled is True
        np.testing.assert_array_equal(adapter.last_commanded_phase, initial_phase)
        assert adapter._presenter is None
        assert frames == []

        adapter.apply_phase(np.zeros(adapter.shape_yx))
        assert np.all(frames[-1][:, :1272] == 6)
        assert np.all(frames[-1][:, 1272:] == 0)
    finally:
        adapter.close()


def test_runtime_correction_enable_only_changes_the_next_apply(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import zlc_atom.devices.slm.device_types as module

    _patch_dvi_without_controller(monkeypatch)
    frames: list[np.ndarray] = []
    monkeypatch.setattr(
        module,
        "_open_dvi_presenter",
        lambda _name: (lambda frame: frames.append(frame.copy()), lambda: None),
    )
    correction_path = tmp_path / "runtime-correction.bmp"
    Image.fromarray(
        np.full((1024, 1272), 9, dtype=np.uint8), mode="L"
    ).save(correction_path)
    adapter = X15213Adapter(_config())

    try:
        adapter.load_correction(correction_path)
        adapter.apply_phase(np.zeros(adapter.shape_yx))
        assert np.all(frames[-1][:, :1272] == 8)
        assert np.all(frames[-1][:, 1272:] == 0)
        commanded = adapter.last_commanded_phase.copy()

        adapter.set_correction_enabled(False)
        assert adapter.correction_enabled is False
        assert len(frames) == 1
        np.testing.assert_array_equal(adapter.last_commanded_phase, commanded)
        adapter.apply_phase(np.zeros(adapter.shape_yx))
        assert np.all(frames[-1] == 0)

        adapter.set_correction_enabled(True)
        assert adapter.correction_enabled is True
        assert len(frames) == 2
        np.testing.assert_array_equal(adapter.last_commanded_phase, commanded)
        adapter.apply_phase(np.zeros(adapter.shape_yx))
        assert np.all(frames[-1][:, :1272] == 8)
        assert np.all(frames[-1][:, 1272:] == 0)
    finally:
        adapter.close()


def test_dvi_applies_orientation_correction_and_active_raster(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import zlc_atom.devices.slm.device_types as module

    _patch_dvi_without_controller(monkeypatch)
    correction = np.zeros((1024, 1272), dtype=np.uint8)
    correction[0, 0] = 255
    correction_path = tmp_path / "correction.bmp"
    Image.fromarray(correction, mode="L").save(correction_path)
    frames: list[np.ndarray] = []
    closed: list[bool] = []
    monkeypatch.setattr(
        module,
        "_open_dvi_presenter",
        lambda _name: (
            lambda frame: frames.append(frame.copy()),
            lambda: closed.append(True),
        ),
    )
    adapter = X15213Adapter(
        _config(
            correction_path=str(correction_path),
            flip_x=True,
        )
    )
    phase = np.zeros(adapter.shape_yx, dtype=np.float64)
    phase[0, 0] = np.pi

    adapter.load_correction(correction_path)
    assert adapter._presenter is None
    commanded = adapter.apply_phase(phase)

    assert commanded.shape == (1024, 1272)
    assert not commanded.flags.writeable
    np.testing.assert_array_equal(commanded, adapter.last_commanded_phase)
    assert len(frames) == 1
    frame = frames[0]
    assert frame.shape == (1024, 1280)
    assert frame.dtype == np.uint8
    assert frame[0, 0] == 224
    assert frame[0, 1271] == 113
    assert np.all(frame[:, 1272:] == 0)
    adapter.close()
    assert closed == [True]


def test_phase_image_codes_then_correction_wrap_then_852_drive_quantization(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import zlc_atom.devices.slm.device_types as module

    _patch_dvi_without_controller(monkeypatch)
    frames: list[np.ndarray] = []
    monkeypatch.setattr(
        module,
        "_open_dvi_presenter",
        lambda _name: (lambda frame: frames.append(frame.copy()), lambda: None),
    )
    adapter = X15213Adapter(_config())
    phase = np.zeros(adapter.shape_yx, dtype=np.float64)
    codes = np.array((0, 64, 128, 192, 255), dtype=np.float64)
    phase[0, : codes.size] = codes * (2.0 * np.pi / 256.0)

    try:
        adapter.apply_phase(phase)
        np.testing.assert_array_equal(
            frames[-1][0, : codes.size], np.array((0, 57, 113, 169, 224))
        )

        correction = np.zeros(adapter.shape_yx, dtype=np.uint8)
        correction[0, 0] = 255
        correction_path = tmp_path / "wrap-at-256.bmp"
        Image.fromarray(correction, mode="L").save(correction_path)
        adapter.load_correction(correction_path)

        adapter.apply_phase(np.zeros(adapter.shape_yx))
        assert frames[-1][0, 0] == 224
        one_code = np.zeros(adapter.shape_yx, dtype=np.float64)
        one_code[0, 0] = 2.0 * np.pi / 256.0
        adapter.apply_phase(one_code)
        assert frames[-1][0, 0] == 0
    finally:
        adapter.close()


def test_correction_filename_scales_unwrapped_785_phase_for_852_and_checks_serial(
    tmp_path: Path,
) -> None:
    import zlc_atom.devices.slm.device_types as module

    x = np.arange(1272, dtype=np.uint16)
    correction = np.broadcast_to(x % 256, (1024, 1272)).astype(np.uint8)
    correction_path = tmp_path / "CAL_LSH0804382_785nm.bmp"
    Image.fromarray(correction, mode="L").save(correction_path)

    converted = module._load_correction(
        str(correction_path),
        expected_serial="LSH0804382",
        wavelength_nm=852.0,
    )
    probes = np.array((0, 1, 64, 255, 256, 512, 1271))
    expected = np.mod(
        np.floor(probes.astype(np.float64) * (785.0 / 852.0) + 0.5),
        256.0,
    ).astype(np.uint8)
    np.testing.assert_array_equal(converted[0, probes], expected)

    wrong_head = tmp_path / "CAL_ANOTHER_HEAD_785nm.bmp"
    Image.fromarray(correction, mode="L").save(wrong_head)
    with pytest.raises(ValueError, match="serial.*profile serial"):
        module._load_correction(
            str(wrong_head),
            expected_serial="LSH0804382",
            wavelength_nm=852.0,
        )


def test_usb_writes_active_gray_reads_it_back_and_keeps_science_phase(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import zlc_atom.devices.slm.device_types as module

    sdk = _UsbSdk()
    monkeypatch.setattr(module, "_find_sdk_directory", lambda _authored="": tmp_path)
    monkeypatch.setattr(module, "_load_sdk", lambda _directory: (sdk, None))
    adapter = X15213Adapter(
        _config(
            transport="usb",
            display_name="",
            serial="  LSH0804382  ",
        )
    )
    phase = np.full(adapter.shape_yx, np.pi, dtype=np.float64)

    commanded = adapter.apply_phase(phase)

    assert np.all(sdk.display == 113)
    np.testing.assert_array_equal(commanded, adapter.last_commanded_phase)
    assert np.allclose(commanded, np.float32(np.pi))
    sdk.bad_readback = True
    with pytest.raises(RuntimeError, match="readback"):
        adapter.apply_phase(np.zeros(adapter.shape_yx))
    np.testing.assert_array_equal(adapter.last_commanded_phase, commanded)
    adapter.close()
    assert sdk.closed


def test_usb_close_failure_retains_board_sdk_and_handle_for_retry(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import zlc_atom.devices.slm.device_types as module

    sdk = _UsbSdk(close_results=(0, 1))
    handle = _Handle()
    monkeypatch.setattr(module, "_find_sdk_directory", lambda _authored="": tmp_path)
    monkeypatch.setattr(module, "_load_sdk", lambda _directory: (sdk, handle))
    adapter = X15213Adapter(
        _config(transport="usb", display_name="", serial="LSH0804382")
    )

    with pytest.raises(RuntimeError, match="Close_Dev"):
        adapter.close()

    assert adapter._board_id == 7
    assert adapter._sdk is sdk
    assert adapter._dll_handle is handle
    assert adapter._closed is False
    assert handle.close_count == 0

    adapter.close()

    assert sdk.close_count == 2
    assert handle.close_count == 1
    assert adapter._board_id is None
    assert adapter._sdk is None
    assert adapter._dll_handle is None
    assert adapter._closed is True


def test_usb_handle_close_failure_retries_handle_without_reclosing_board(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import zlc_atom.devices.slm.device_types as module

    sdk = _UsbSdk()
    handle = _Handle(failures=1)
    monkeypatch.setattr(module, "_find_sdk_directory", lambda _authored="": tmp_path)
    monkeypatch.setattr(module, "_load_sdk", lambda _directory: (sdk, handle))
    adapter = X15213Adapter(
        _config(transport="usb", display_name="", serial="LSH0804382")
    )

    with pytest.raises(RuntimeError, match="handle close failed"):
        adapter.close()

    assert sdk.close_count == 1
    assert adapter._board_id is None
    assert adapter._sdk is sdk
    assert adapter._dll_handle is handle
    assert adapter._closed is False

    adapter.close()

    assert sdk.close_count == 1
    assert handle.close_count == 2
    assert adapter._sdk is None
    assert adapter._dll_handle is None
    assert adapter._closed is True


def test_usb_dvi_mode_switch_reboots_and_reopens_in_one_initialization(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import zlc_atom.devices.slm.device_types as module

    sdk = _UsbSdk(mode=0)
    monkeypatch.setattr(module, "_find_sdk_directory", lambda _authored="": tmp_path)
    monkeypatch.setattr(module, "_load_sdk", lambda _directory: (sdk, None))
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    adapter = X15213Adapter(
        _config(transport="usb", display_name="", serial="LSH0804382")
    )
    assert sdk.selected == 1
    assert sdk.rebooted
    assert adapter.identity.endswith(":LSH0804382")
    adapter.close()


def test_usb_serial_binding_is_exact_after_stripping_and_closes_on_mismatch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import zlc_atom.devices.slm.device_types as module

    sdk = _UsbSdk()
    sdk.serial = b"LSH0804383"
    handle = _Handle()
    monkeypatch.setattr(module, "_find_sdk_directory", lambda _authored="": tmp_path)
    monkeypatch.setattr(module, "_load_sdk", lambda _directory: (sdk, handle))

    with pytest.raises(RuntimeError, match="requested X15213 serial"):
        X15213Adapter(
            _config(transport="usb", display_name="", serial="  LSH0804382  ")
        )

    assert sdk.close_count == 1
    assert handle.close_count == 1


@pytest.mark.parametrize(
    ("serial_result", "mode_result", "operation"),
    ((0, 1, "Check_HeadSerial"), (1, 0, "Mode_Check")),
)
def test_usb_first_probe_failure_always_closes_the_open_board(
    monkeypatch,
    tmp_path: Path,
    serial_result: int,
    mode_result: int,
    operation: str,
) -> None:
    import zlc_atom.devices.slm.device_types as module

    sdk = _UsbSdk(serial_result=serial_result, mode_result=mode_result)
    handle = _Handle()
    monkeypatch.setattr(module, "_find_sdk_directory", lambda _authored="": tmp_path)
    monkeypatch.setattr(module, "_load_sdk", lambda _directory: (sdk, handle))

    with pytest.raises(RuntimeError, match=operation):
        X15213Adapter(
            _config(transport="usb", display_name="", serial="LSH0804382")
        )

    assert sdk.close_count == 1
    assert handle.close_count == 1


def test_dvi_initialization_normalizes_display_and_marks_endpoint_only_evidence(
    monkeypatch,
) -> None:
    import zlc_atom.devices.slm.device_types as module

    _patch_dvi_without_controller(monkeypatch)
    adapter = X15213Adapter(_config(display_name=r"  \\.\display2  "))
    try:
        assert adapter.identity == r"hamamatsu-x15213:dvi-display:\\.\DISPLAY2"
        assert adapter._dvi_controller_mode_proven is False
    finally:
        adapter.close()


def test_dvi_initialization_switches_a_connected_controller_to_dvi_mode(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import zlc_atom.devices.slm.device_types as module

    sdk = _UsbSdk(mode=1)
    handle = _Handle()
    monkeypatch.setattr(module, "_windows_displays", lambda: (_display_endpoint(),))
    monkeypatch.setattr(module, "_find_sdk_directory", lambda _authored="": tmp_path)
    monkeypatch.setattr(module, "_load_sdk", lambda _directory: (sdk, handle))

    adapter = X15213Adapter(_config())
    try:
        assert sdk.selected == 0
        assert sdk.rebooted
        assert sdk.close_count == 1
        assert handle.close_count == 1
        assert adapter._dvi_controller_mode_proven is True
    finally:
        adapter.close()


def test_dvi_unknown_connected_controller_mode_is_rejected_and_closed(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import zlc_atom.devices.slm.device_types as module

    sdk = _UsbSdk(mode=7)
    handle = _Handle()
    monkeypatch.setattr(module, "_windows_displays", lambda: (_display_endpoint(),))
    monkeypatch.setattr(module, "_find_sdk_directory", lambda _authored="": tmp_path)
    monkeypatch.setattr(module, "_load_sdk", lambda _directory: (sdk, handle))

    with pytest.raises(RuntimeError, match="unknown controller mode 7"):
        X15213Adapter(_config())

    assert sdk.close_count == 1
    assert handle.close_count == 1


def test_dvi_sdk_without_a_usb_controller_keeps_endpoint_only_evidence(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import zlc_atom.devices.slm.device_types as module

    class _NoUsb:
        def Open_Dev(self, _ids, _size):
            return 0

    handle = _Handle()
    monkeypatch.setattr(module, "_windows_displays", lambda: (_display_endpoint(),))
    monkeypatch.setattr(module, "_find_sdk_directory", lambda _authored="": tmp_path)
    monkeypatch.setattr(module, "_load_sdk", lambda _directory: (_NoUsb(), handle))

    adapter = X15213Adapter(_config())
    try:
        assert adapter._dvi_controller_mode_proven is False
        assert adapter.identity.startswith("hamamatsu-x15213:dvi-display:")
        assert handle.close_count == 1
    finally:
        adapter.close()


@pytest.mark.parametrize(
    ("serial_result", "mode_result", "operation"),
    ((0, 1, "Check_HeadSerial"), (1, 0, "Mode_Check")),
)
def test_dvi_controller_probe_failure_always_closes_the_open_board(
    monkeypatch,
    tmp_path: Path,
    serial_result: int,
    mode_result: int,
    operation: str,
) -> None:
    import zlc_atom.devices.slm.device_types as module

    sdk = _UsbSdk(serial_result=serial_result, mode_result=mode_result)
    handle = _Handle()
    monkeypatch.setattr(module, "_windows_displays", lambda: (_display_endpoint(),))
    monkeypatch.setattr(module, "_find_sdk_directory", lambda _authored="": tmp_path)
    monkeypatch.setattr(module, "_load_sdk", lambda _directory: (sdk, handle))

    with pytest.raises(RuntimeError, match=operation):
        X15213Adapter(_config())

    assert sdk.close_count == 1
    assert handle.close_count == 1


def test_dvi_native_client_check_uses_per_monitor_v2_physical_coordinates(
    monkeypatch,
) -> None:
    import zlc_atom.devices.slm.device_types as module

    class _User32:
        def __init__(self) -> None:
            self.dpi_contexts: list[int] = []

        def SetThreadDpiAwarenessContext(self, context):
            self.dpi_contexts.append(ctypes.c_ssize_t(context.value).value)
            return 1

        def GetClientRect(self, hwnd, target):
            assert int(hwnd.value) == 41
            rect = ctypes.cast(target, ctypes.POINTER(ctypes.wintypes.RECT)).contents
            rect.left, rect.top, rect.right, rect.bottom = 0, 0, 1280, 1024
            return 1

        def ClientToScreen(self, hwnd, target):
            assert int(hwnd.value) == 41
            point = ctypes.cast(target, ctypes.POINTER(ctypes.wintypes.POINT)).contents
            point.x, point.y = 1920, 0
            return 1

    user32 = _User32()
    monkeypatch.setattr(module.ctypes, "windll", SimpleNamespace(user32=user32))

    module._set_dvi_thread_dpi_awareness()
    actual = module._native_dvi_client_geometry(41)

    assert user32.dpi_contexts == [-4]
    assert actual == (1920, 0, 1280, 1024)
