from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from zlc_atom.devices.slm import SlmAdapter
from zlc_atom.devices.slm.device_types import (
    DEVICE_TYPES,
    HAMAMATSU_X15213_SCHEMA,
    X15213Adapter,
    _discover_x15213,
    _load_correction,
    _load_profile,
)
from zlc_atom.install import create_installation


class _UsbSdk:
    def __init__(self, *, mode: int = 1, serial: bytes = b"LSH0804382") -> None:
        self.mode = mode
        self.serial = serial
        self.display = np.zeros((1024, 1272), dtype=np.uint8)
        self.open_count = 0
        self.close_count = 0
        self.write_count = 0
        self.selected: int | None = None
        self.rebooted = False
        self.write_result = 1
        self.write_updates = True
        self.change_result = 1
        self.check_result = 1
        self.bad_readback = False
        self.close_results: list[int] = []

    def Open_Dev(self, ids, _size):
        self.open_count += 1
        ids[0] = 7
        return 1

    def Check_HeadSerial(self, _board, target, _size):
        target.value = self.serial
        return 1

    def Mode_Check(self, _board, target):
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
        self.write_count += 1
        if self.write_updates:
            self.display = np.ctypeslib.as_array(
                source, shape=(int(size),)
            ).reshape(int(height), int(width)).copy()
        return self.write_result

    def Change_DispSlot(self, _board, _slot):
        return self.change_result

    def Check_Disp_IMG(self, _board, size, _width, _height, target):
        if self.check_result != 1:
            return self.check_result
        observed = self.display.copy()
        if self.bad_readback:
            observed[0, 0] ^= np.uint8(1)
        np.ctypeslib.as_array(target, shape=(int(size),))[:] = observed.reshape(-1)
        return 1

    def Close_Dev(self, _ids, _size):
        self.close_count += 1
        return self.close_results.pop(0) if self.close_results else 1


class _Handle:
    def __init__(self) -> None:
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


def _config(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "sdk_directory": "",
        "device_profile": "LSH0804382",
        "wavelength_nm": 852.0,
        "correction_path": "",
        "flip_x": False,
        "flip_y": False,
    }
    values.update(changes)
    return values


def _patch_usb(monkeypatch, sdk: _UsbSdk, handle: _Handle | None = None) -> _Handle:
    import zlc_atom.devices.slm.device_types as module

    result = handle or _Handle()
    monkeypatch.setattr(module, "_find_sdk_directory", lambda _authored="": Path("sdk"))
    monkeypatch.setattr(module, "_load_sdk", lambda _directory: (sdk, result))
    return result


def test_x15213_product_descriptor_and_schema_are_usb_only() -> None:
    assert len(DEVICE_TYPES) == 1
    descriptor = DEVICE_TYPES[0]
    assert descriptor.type_id == "slm.hamamatsu_x15213"
    assert descriptor.domain == "slm"
    assert descriptor.capabilities == ("slm.phase",)
    assert descriptor.discover is _discover_x15213
    assert descriptor.control_factory is not None
    assert set(HAMAMATSU_X15213_SCHEMA.field_names) == set(_config())


def test_profile_is_strict_and_records_physical_provenance_boundaries(
    monkeypatch, tmp_path: Path
) -> None:
    import zlc_atom.devices.slm.device_types as module

    profile = _load_profile("LSH0804382")
    assert profile["model"] == "X15213 (exact type suffix not recorded)"
    assert profile["serial"] == "LSH0804382"
    assert profile["default_wavelength_nm"] == 852.0
    assert profile["phase_curve_wavelength_nm"] == 785.0
    assert "not recorded" in str(profile["phase_curve_source"])
    assert profile["settle_seconds"] == 0.05
    assert "pending" in str(profile["settle_source"])
    assert np.asarray(profile["phase_pi_by_gray"]).shape == (256,)

    payload_path = (
        Path(module.__file__).resolve().parent / "profiles" / "LSH0804382.json"
    )
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    duplicate = payload_path.read_text(encoding="utf-8").replace(
        '"serial": "LSH0804382",',
        '"serial": "LSH0804382", "serial": "OTHER",',
    )
    (tmp_path / "duplicate.json").write_text(duplicate, encoding="utf-8")
    payload["settle_seconds"] = "0.05"
    (tmp_path / "coerced.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(module, "_PROFILE_DIRECTORY", tmp_path)
    with pytest.raises(ValueError, match="strict JSON"):
        _load_profile("duplicate")
    with pytest.raises(ValueError, match="settle_seconds"):
        _load_profile("coerced")


def test_real_installation_starts_unknown_without_fabricating_zero(
    monkeypatch,
) -> None:
    sdk = _UsbSdk()
    handle = _patch_usb(monkeypatch, sdk)
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
        assert isinstance(slm, SlmAdapter)
        assert slm.last_commanded_phase is None
        assert slm.command_revision == 0
        assert slm.mapping_revision == 0
        assert slm.last_command_receipt == {
            "transport": "usb",
            "identity": "hamamatsu-x15213:usb:LSH0804382",
            "profile": "LSH0804382",
            "model": "X15213 (exact type suffix not recorded)",
            "serial": "LSH0804382",
            "wavelength_nm": 852.0,
            "flip_x": False,
            "flip_y": False,
            "correction_path": "",
            "correction_enabled": False,
            "mapping_revision": 0,
            "settle_seconds": 0.05,
            "settle_source": "Repository default; optical settle acceptance pending",
            "phase_curve_source": (
                "Repository calibration values; measurement provenance not recorded"
            ),
            "outcome": "unknown",
            "command_revision": 0,
            "stage": "uncommanded",
            "readback": "not-run",
        }
        assert sdk.write_count == 0
    finally:
        installation.close()
    assert sdk.close_count == 1
    assert handle.close_count == 1


def test_successful_usb_command_is_known_only_after_readback_and_settle(
    monkeypatch,
) -> None:
    sdk = _UsbSdk()
    _patch_usb(monkeypatch, sdk)
    adapter = X15213Adapter(_config(flip_x=True, flip_y=True))
    try:
        phase = np.full(adapter.shape_yx, np.pi, dtype=np.float32)
        commanded = adapter.apply_phase(phase)
        assert not commanded.flags.writeable
        np.testing.assert_array_equal(adapter.last_commanded_phase, commanded)
        assert sdk.write_count == 1
        assert adapter.command_revision == 1
        receipt = adapter.last_command_receipt
        assert receipt["outcome"] == "known-new"
        assert receipt["stage"] == "complete"
        assert receipt["readback"] == "matched-new"
        assert receipt["command_revision"] == 1
        assert receipt["mapping_revision"] == 0
        assert receipt["transport"] == "usb"
    finally:
        adapter.close()


def test_usb_failure_outcomes_preserve_old_or_become_unknown(
    monkeypatch,
) -> None:
    sdk = _UsbSdk()
    _patch_usb(monkeypatch, sdk)
    adapter = X15213Adapter(_config())
    old = adapter.apply_phase(np.zeros(adapter.shape_yx, dtype=np.float32))

    sdk.write_updates = False
    sdk.write_result = 0
    with pytest.raises(RuntimeError, match="Write_FMemArray"):
        adapter.apply_phase(np.full(adapter.shape_yx, np.pi, dtype=np.float32))
    assert adapter.last_command_receipt["outcome"] == "known-old"
    assert adapter.last_command_receipt["readback"] == "matched-old"
    np.testing.assert_array_equal(adapter.last_commanded_phase, old)

    sdk.write_updates = True
    sdk.write_result = 1
    sdk.change_result = 0
    with pytest.raises(RuntimeError, match="Change_DispSlot"):
        adapter.apply_phase(np.full(adapter.shape_yx, np.pi / 2.0, dtype=np.float32))
    assert adapter.last_command_receipt["outcome"] == "unknown"
    assert adapter.last_command_receipt["stage"] == "display"
    assert adapter.last_command_receipt["readback"] == "matched-new"
    assert adapter.last_commanded_phase is None

    sdk.change_result = 1
    sdk.check_result = 0
    with pytest.raises(RuntimeError, match="Check_Disp_IMG"):
        adapter.apply_phase(np.full(adapter.shape_yx, np.pi, dtype=np.float32))
    assert adapter.last_commanded_phase is None
    assert adapter.last_command_receipt["outcome"] == "unknown"
    assert adapter.last_command_receipt["stage"] == "readback"
    assert adapter.command_revision == 4
    adapter.close()


def test_settle_failure_clears_command_knowledge(monkeypatch) -> None:
    import zlc_atom.devices.slm.device_types as module

    sdk = _UsbSdk()
    _patch_usb(monkeypatch, sdk)
    adapter = X15213Adapter(_config())

    def fail_settle(_seconds: float) -> None:
        raise RuntimeError("settle interrupted")

    monkeypatch.setattr(module.time, "sleep", fail_settle)
    with pytest.raises(RuntimeError, match="settle interrupted"):
        adapter.apply_phase(np.zeros(adapter.shape_yx, dtype=np.float32))
    assert adapter.last_commanded_phase is None
    assert adapter.last_command_receipt["outcome"] == "unknown"
    assert adapter.last_command_receipt["stage"] == "settle"
    assert adapter.last_command_receipt["readback"] == "matched-new"
    adapter.close()


def test_correction_revision_is_atomic_and_command_receipt_freezes_mapping(
    monkeypatch, tmp_path: Path
) -> None:
    sdk = _UsbSdk()
    _patch_usb(monkeypatch, sdk)
    correction_path = tmp_path / "CAL_LSH0804382_852nm.bmp"
    correction = np.zeros((1024, 1272), dtype=np.uint8)
    correction[0, 0] = 255
    Image.fromarray(correction, mode="L").save(correction_path)

    adapter = X15213Adapter(_config())
    try:
        assert adapter.load_correction(correction_path) == 1
        assert adapter.mapping_revision == 1
        adapter.apply_phase(np.zeros(adapter.shape_yx, dtype=np.float32))
        frozen = adapter.last_command_receipt
        assert frozen["mapping_revision"] == 1
        assert frozen["correction_enabled"] is True
        assert frozen["correction_path"] == str(correction_path.resolve())

        assert adapter.set_correction_enabled(False) == 2
        assert adapter.set_correction_enabled(False) == 2
        assert adapter.mapping_revision == 2
        assert adapter.last_command_receipt == frozen
        adapter.apply_phase(np.zeros(adapter.shape_yx, dtype=np.float32))
        assert adapter.last_command_receipt["mapping_revision"] == 2
        assert adapter.last_command_receipt["correction_enabled"] is False

        assert adapter.load_correction(correction_path) == 3
        previous_phase = adapter.last_commanded_phase
        previous_receipt = adapter.last_command_receipt
        sdk.write_result = 0
        sdk.write_updates = False
        with pytest.raises(RuntimeError, match="Write_FMemArray"):
            adapter.apply_phase(np.zeros(adapter.shape_yx, dtype=np.float32))
        assert adapter.last_command_receipt["readback"] == "matched-old"
        assert adapter.last_command_receipt["outcome"] == "known-old"
        assert adapter.last_command_receipt["mapping_revision"] == 2
        assert adapter.last_command_receipt["correction_enabled"] is False
        assert adapter.last_command_receipt["correction_path"] == str(
            correction_path.resolve()
        )
        assert adapter.last_command_receipt["command_revision"] == 3
        assert adapter.mapping_revision == 3
        assert previous_receipt["mapping_revision"] == 2
        np.testing.assert_array_equal(adapter.last_commanded_phase, previous_phase)
    finally:
        adapter.close()


def test_correction_rejects_unproven_cross_wavelength_conversion(
    tmp_path: Path,
) -> None:
    values = np.zeros((1024, 1272), dtype=np.uint8)
    wrong_wavelength = tmp_path / "CAL_LSH0804382_785nm.bmp"
    Image.fromarray(values, mode="L").save(wrong_wavelength)
    with pytest.raises(ValueError, match="two-dimensional phase-unwrapping evidence"):
        _load_correction(
            str(wrong_wavelength),
            expected_serial="LSH0804382",
            wavelength_nm=852.0,
        )
    wrong_serial = tmp_path / "CAL_OTHER_852nm.bmp"
    Image.fromarray(values, mode="L").save(wrong_serial)
    with pytest.raises(ValueError, match="serial"):
        _load_correction(
            str(wrong_serial),
            expected_serial="LSH0804382",
            wavelength_nm=852.0,
        )


def test_discovery_is_usb_only_and_does_not_send_a_phase(monkeypatch) -> None:
    sdk = _UsbSdk()
    handle = _patch_usb(monkeypatch, sdk)
    found = _discover_x15213()
    assert len(found) == 1
    candidate = found[0]
    assert candidate.instance_id == "x15213_usb_LSH0804382"
    assert candidate.type_id == "slm.hamamatsu_x15213"
    assert set(candidate.parameters) == set(_config())
    assert candidate.parameters["device_profile"] == "LSH0804382"
    assert candidate.parameters["wavelength_nm"] == 852.0
    assert sdk.write_count == 0
    assert sdk.close_count == 1
    assert handle.close_count == 1

    import zlc_atom.devices.slm.device_types as module

    monkeypatch.setattr(module, "_find_sdk_directory", lambda _authored="": None)
    assert _discover_x15213() == ()


def test_usb_mode_switch_reboots_reopens_and_rechecks_identity(monkeypatch) -> None:
    sdk = _UsbSdk(mode=0)
    _patch_usb(monkeypatch, sdk)
    adapter = X15213Adapter(_config())
    try:
        assert sdk.selected == 1
        assert sdk.rebooted is True
        assert sdk.open_count == 2
        assert sdk.close_count == 1
        assert adapter.identity == "hamamatsu-x15213:usb:LSH0804382"
    finally:
        adapter.close()
    assert sdk.close_count == 2

    wrong = _UsbSdk(serial=b"OTHER")
    handle = _patch_usb(monkeypatch, wrong)
    with pytest.raises(RuntimeError, match="profile serial"):
        X15213Adapter(_config())
    assert wrong.close_count == 1
    assert handle.close_count == 1


def test_usb_close_failure_is_visible_and_retryable(monkeypatch) -> None:
    sdk = _UsbSdk()
    sdk.close_results = [0, 1]
    handle = _patch_usb(monkeypatch, sdk)
    adapter = X15213Adapter(_config())
    with pytest.raises(RuntimeError, match="Close_Dev"):
        adapter.close()
    assert handle.close_count == 0
    adapter.close()
    assert sdk.close_count == 2
    assert handle.close_count == 1
