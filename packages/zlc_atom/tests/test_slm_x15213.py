from __future__ import annotations

import json
from pathlib import Path
import socket
from threading import Barrier, Thread

import numpy as np
import pytest
from PIL import Image

from zlc_atom.devices.slm import SlmAdapter
from zlc_atom.devices.slm.device import _RemoteSlmAdapter, _open_slm_server
from zlc_atom.devices.slm.device_types import (
    DEVICE_TYPES,
    HAMAMATSU_X15213_SCHEMA,
    X15213_SERVER_SCHEMA,
    X15213Adapter,
    _find_sdk_directory,
    _load_sdk,
    _load_correction,
    _load_profile,
    _print_client_endpoints,
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
        "transport": "usb",
        "display_name": "",
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


def _running_server(adapter: SlmAdapter):
    server = _open_slm_server(adapter, "127.0.0.1", 0)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    return server, worker


def test_real_slm_descriptor_matches_the_pulse_server_endpoint_model() -> None:
    assert len(DEVICE_TYPES) == 1
    descriptor = DEVICE_TYPES[0]
    assert descriptor.type_id == "slm.hamamatsu_x15213"
    assert descriptor.domain == "slm"
    assert descriptor.capabilities == ("slm.phase",)
    assert descriptor.discover is None
    assert descriptor.control_factory is not None
    assert HAMAMATSU_X15213_SCHEMA.field_names == ("host", "port")
    assert HAMAMATSU_X15213_SCHEMA.project_values({}) == {
        "host": "127.0.0.1",
        "port": 18862,
    }
    assert set(X15213_SERVER_SCHEMA.field_names) == set(_config())


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
    payload["settle_seconds"] = 0.05
    payload["unexpected"] = 2
    (tmp_path / "unknown-field.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(module, "_PROFILE_DIRECTORY", tmp_path)
    with pytest.raises(ValueError, match="strict JSON"):
        _load_profile("duplicate")
    with pytest.raises(ValueError, match="settle_seconds"):
        _load_profile("coerced")
    with pytest.raises(ValueError, match="invalid field set"):
        _load_profile("unknown-field")


def test_real_installation_dials_its_server_endpoint_and_starts_unknown(
    monkeypatch,
) -> None:
    sdk = _UsbSdk()
    handle = _patch_usb(monkeypatch, sdk)
    physical = X15213Adapter(_config())
    server, worker = _running_server(physical)
    installation = None
    try:
        installation = create_installation(
            (
                {
                    "key": "slm",
                    "type_id": "slm.hamamatsu_x15213",
                    "config": {
                        "host": "127.0.0.1",
                        "port": server.server_address[1],
                    },
                },
            )
        )
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
            "dvi_controller_mode_proven": False,
            "outcome": "unknown",
            "command_revision": 0,
            "stage": "uncommanded",
            "readback": "not-run",
        }
        assert sdk.write_count == 0
    finally:
        if installation is not None:
            installation.close()
        server.shutdown()
        server.server_close()
        worker.join(timeout=2.0)
        physical.close()
    assert not worker.is_alive()
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


def test_sdk_loading_does_not_require_a_second_dll_preflight(
    monkeypatch, tmp_path: Path
) -> None:
    import zlc_atom.devices.slm.device_types as module

    primary = tmp_path / "hpkSLMdaLV.dll"
    primary.write_bytes(b"vendor library placeholder")
    assert _find_sdk_directory(str(tmp_path)) == tmp_path.resolve()

    loaded: list[str] = []
    sentinel = object()
    monkeypatch.setattr(
        module.ctypes,
        "WinDLL",
        lambda library: loaded.append(str(library)) or sentinel,
        raising=False,
    )
    sdk, handle = _load_sdk(None)
    assert sdk is sentinel
    assert handle is None
    assert loaded == ["hpkSLMdaLV.dll"]


def test_dvi_server_transport_needs_no_vendor_dll_and_preserves_the_raster_path(
    monkeypatch,
) -> None:
    import zlc_atom.devices.slm.device_types as module

    endpoint = {
        "name": r"\\.\DISPLAY2",
        "attached": True,
        "primary": False,
        "width": 1280,
        "height": 1024,
        "frequency": 60,
        "x": 1920,
        "y": 0,
    }
    frames: list[np.ndarray] = []
    closed: list[bool] = []
    monkeypatch.setattr(module, "_windows_displays", lambda: (endpoint,))
    monkeypatch.setattr(module, "_prepare_dvi_controller", lambda *_args: False)
    monkeypatch.setattr(
        module,
        "_load_sdk",
        lambda *_args: (_ for _ in ()).throw(AssertionError("DVI loaded the SDK")),
    )
    monkeypatch.setattr(
        module,
        "_open_dvi_presenter",
        lambda _name: (
            lambda frame: frames.append(frame.copy()),
            lambda: closed.append(True),
        ),
    )

    adapter = X15213Adapter(_config(transport="dvi"))
    server, worker = _running_server(adapter)
    installation = None
    try:
        assert adapter.identity == r"hamamatsu-x15213:dvi-display:\\.\DISPLAY2"
        installation = create_installation(
            (
                {
                    "key": "slm",
                    "type_id": "slm.hamamatsu_x15213",
                    "config": {
                        "host": "127.0.0.1",
                        "port": server.server_address[1],
                    },
                },
            )
        )
        assert installation.failures == {}
        remote = installation.capability("slm.phase", key="slm")
        commanded = remote.apply_phase(
            np.full(adapter.shape_yx, np.pi, dtype=np.float32)
        )
        assert len(frames) == 1
        assert frames[0].shape == (1024, 1280)
        assert np.all(frames[0][:, 1272:] == 0)
        np.testing.assert_array_equal(commanded, remote.last_commanded_phase)
        np.testing.assert_array_equal(commanded, adapter.last_commanded_phase)
        assert adapter.last_command_receipt["transport"] == "dvi"
        assert adapter.last_command_receipt["outcome"] == "known-new"
        assert adapter.last_command_receipt["readback"] == "presenter-ack"
        from zlc_atom.devices.slm.solver import _command_receipt

        assert _command_receipt(adapter.last_command_receipt)["transport"] == "dvi"
    finally:
        if installation is not None:
            installation.close()
        server.shutdown()
        server.server_close()
        worker.join(timeout=2.0)
        adapter.close()
    assert not worker.is_alive()
    assert closed == [True]


def test_broken_or_missing_usb_sdk_cannot_block_the_default_dvi_transport(
    monkeypatch,
) -> None:
    import zlc_atom.devices.slm.device_types as module

    monkeypatch.setattr(
        module, "_find_sdk_directory", lambda _authored="": Path("sdk")
    )
    monkeypatch.setattr(
        module,
        "_load_sdk",
        lambda _directory: (_ for _ in ()).throw(
            OSError("could not find hpkSLMdaLV.dll")
        ),
    )
    assert module._prepare_dvi_controller("", "LSH0804382") is False


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


def test_remote_slm_caches_reads_and_only_calls_the_server_to_send_phase(
    monkeypatch,
) -> None:
    import zlc_atom.devices.slm.device as device_module

    sdk = _UsbSdk()
    _patch_usb(monkeypatch, sdk)
    physical = X15213Adapter(_config())
    server, worker = _running_server(physical)
    calls: list[str] = []
    original = device_module._rpc_call

    def counted(endpoint, method, arguments, timeout):
        calls.append(method)
        return original(endpoint, method, arguments, timeout)

    monkeypatch.setattr(device_module, "_rpc_call", counted)
    installation = None
    try:
        installation = create_installation(
            (
                {
                    "key": "slm",
                    "type_id": "slm.hamamatsu_x15213",
                    "config": {
                        "host": "127.0.0.1",
                        "port": server.server_address[1],
                    },
                },
            )
        )
        assert installation.failures == {}
        remote = installation.capability("slm.phase", key="slm")
        assert calls == ["describe"]
        assert remote.identity == physical.identity
        assert remote.shape_yx == physical.shape_yx
        assert remote.last_commanded_phase is None
        assert remote.command_revision == 0
        assert remote.mapping_revision == 0
        assert remote.last_command_receipt["outcome"] == "unknown"
        assert calls == ["describe"]

        expected = np.full(remote.shape_yx, np.pi / 3.0, dtype=np.float32)
        applied = remote.apply_phase(expected)
        assert calls == ["describe", "apply"]
        assert sdk.write_count == 1
        assert remote.command_revision == 1
        assert remote.last_command_receipt["outcome"] == "known-new"
        np.testing.assert_array_equal(applied, remote.last_commanded_phase)
        np.testing.assert_array_equal(applied, physical.last_commanded_phase)

        timed_out = False

        def timeout_once(endpoint, method, arguments, timeout):
            nonlocal timed_out
            calls.append(method)
            if method == "apply" and not timed_out:
                timed_out = True
                raise socket.timeout("simulated reply timeout")
            return original(endpoint, method, arguments, timeout)

        monkeypatch.setattr(device_module, "_rpc_call", timeout_once)
        with pytest.raises(socket.timeout):
            remote.apply_phase(
                np.full(remote.shape_yx, np.pi / 2.0, dtype=np.float32)
            )
        assert remote.last_commanded_phase is None
        assert remote.last_command_receipt["outcome"] == "unknown"
        recovered = remote.apply_phase(
            np.full(remote.shape_yx, np.pi / 2.0, dtype=np.float32)
        )
        assert calls == ["describe", "apply", "apply", "describe", "apply"]
        assert sdk.write_count == 2
        np.testing.assert_array_equal(remote.last_commanded_phase, recovered)

    finally:
        if installation is not None:
            installation.close()
        server.shutdown()
        server.server_close()
        worker.join(timeout=2.0)
        physical.close()
    assert not worker.is_alive()
    assert calls == ["describe", "apply", "apply", "describe", "apply"]
    with pytest.raises(RuntimeError, match="closed"):
        remote.apply_phase(expected)
    assert calls == ["describe", "apply", "apply", "describe", "apply"]


def test_remote_slm_rejects_a_stale_writer_and_refreshes_physical_truth(
    monkeypatch,
) -> None:
    sdk = _UsbSdk()
    _patch_usb(monkeypatch, sdk)
    physical = X15213Adapter(_config())
    server, worker = _running_server(physical)
    first = second = None
    try:
        first = _RemoteSlmAdapter("127.0.0.1", server.server_address[1], 2.0)
        second = _RemoteSlmAdapter("127.0.0.1", server.server_address[1], 2.0)
        barrier = Barrier(3)
        results: dict[str, np.ndarray] = {}
        errors: dict[str, BaseException] = {}

        def send(name: str, adapter, value: float) -> None:
            barrier.wait()
            try:
                results[name] = adapter.apply_phase(
                    np.full(adapter.shape_yx, value, dtype=np.float32)
                )
            except BaseException as error:
                errors[name] = error

        workers = (
            Thread(target=send, args=("first", first, np.pi / 4.0)),
            Thread(target=send, args=("second", second, np.pi / 2.0)),
        )
        for command in workers:
            command.start()
        barrier.wait()
        for command in workers:
            command.join(timeout=3.0)
            assert not command.is_alive()

        assert len(results) == len(errors) == 1
        assert "stale SLM command" in str(next(iter(errors.values())))
        winner_name, phase1 = next(iter(results.items()))
        loser = second if winner_name == "first" else first
        assert loser.command_revision == 1
        np.testing.assert_array_equal(loser.last_commanded_phase, phase1)
        assert sdk.write_count == 1

        phase2 = loser.apply_phase(
            np.full(loser.shape_yx, 3.0 * np.pi / 4.0, dtype=np.float32)
        )
        assert loser.command_revision == 2
        np.testing.assert_array_equal(physical.last_commanded_phase, phase2)
        assert sdk.write_count == 2
    finally:
        if first is not None:
            first.close()
        if second is not None:
            second.close()
        server.shutdown()
        server.server_close()
        worker.join(timeout=2.0)
        physical.close()
    assert not worker.is_alive()


def test_remote_slm_preserves_declared_failures_and_marks_bad_replies_unknown(
    monkeypatch,
) -> None:
    import zlc_atom.devices.slm.device as device_module

    sdk = _UsbSdk()
    _patch_usb(monkeypatch, sdk)
    physical = X15213Adapter(_config())
    server, worker = _running_server(physical)
    remote = None
    original = device_module._rpc_call
    try:
        for request in (
            {"version": True, "method": "describe"},
            {
                "version": 1,
                "method": "apply",
                "command_revision": 0,
                "mapping_revision": 0,
                "shape_yx": [1024.0, 1272.0],
            },
        ):
            with socket.create_connection(server.server_address, timeout=2.0) as invalid:
                device_module._send_packet(invalid, request)
                reply, _payload = device_module._recv_packet(invalid)
            assert reply["ok"] is False
        assert sdk.write_count == 0
        with socket.create_connection(server.server_address, timeout=2.0) as oversized:
            oversized.sendall(
                device_module._REMOTE_HEADER.pack(
                    device_module._MAX_REMOTE_METADATA_BYTES + 1, 0
                )
            )
            assert oversized.recv(1) == b""
        remote = _RemoteSlmAdapter("127.0.0.1", server.server_address[1], 2.0)
        old = remote.apply_phase(
            np.full(remote.shape_yx, np.pi / 4.0, dtype=np.float32)
        )
        sdk.write_updates = False
        sdk.write_result = 0
        with pytest.raises(RuntimeError, match="Write_FMemArray"):
            remote.apply_phase(
                np.full(remote.shape_yx, np.pi / 2.0, dtype=np.float32)
            )
        assert remote.last_command_receipt["outcome"] == "known-old"
        np.testing.assert_array_equal(remote.last_commanded_phase, old)

        sdk.write_updates = True
        sdk.write_result = 1

        def malformed_after_apply(endpoint, method, arguments, timeout):
            metadata, payload = original(endpoint, method, arguments, timeout)
            if method == "apply":
                metadata["version"] = True
            return metadata, payload

        monkeypatch.setattr(device_module, "_rpc_call", malformed_after_apply)
        with pytest.raises(ValueError, match="protocol version"):
            remote.apply_phase(
                np.full(remote.shape_yx, 3.0 * np.pi / 4.0, dtype=np.float32)
            )
        assert physical.command_revision == 3
        assert remote.last_commanded_phase is None
        assert remote.last_command_receipt["outcome"] == "unknown"

        monkeypatch.setattr(device_module, "_rpc_call", original)
        recovered = remote.apply_phase(
            np.full(remote.shape_yx, np.pi, dtype=np.float32)
        )
        assert remote.command_revision == 4
        np.testing.assert_array_equal(remote.last_commanded_phase, recovered)

        sdk.change_result = 0
        with pytest.raises(RuntimeError, match="Change_DispSlot"):
            remote.apply_phase(
                np.full(remote.shape_yx, 5.0 * np.pi / 4.0, dtype=np.float32)
            )
        assert remote.command_revision == 5
        assert remote.last_commanded_phase is None
        assert remote.last_command_receipt["outcome"] == "unknown"
    finally:
        if remote is not None:
            remote.close()
        server.shutdown()
        server.server_close()
        worker.join(timeout=2.0)
        physical.close()
    assert not worker.is_alive()


def test_remote_packet_grammar_rejects_partial_duplicate_and_nonfinite_input(
    monkeypatch,
) -> None:
    import zlc_atom.devices.slm.device as device_module

    def huge_shape(_endpoint, _method, _arguments, _timeout):
        return {
            "version": 1,
            "ok": True,
            "error": None,
            "state": {
                "identity": "huge-simulated-slm",
                "shape_yx": [10**10, 10**10],
                "command_revision": 0,
                "mapping_revision": 0,
                "receipt": {
                    "identity": "huge-simulated-slm",
                    "outcome": "unknown",
                    "command_revision": 0,
                    "mapping_revision": 0,
                },
                "phase_bytes": 0,
            },
        }, b""

    monkeypatch.setattr(device_module, "_rpc_call", huge_shape)
    with pytest.raises(ValueError, match="payload bound"):
        _RemoteSlmAdapter("127.0.0.1", 1, 1.0)

    def invalid_state(_endpoint, _method, _arguments, _timeout):
        phase = np.full((2, 2), 7.0 * np.pi, dtype="<f4").tobytes()
        return {
            "version": 1,
            "ok": True,
            "error": None,
            "state": {
                "identity": "broken-simulated-slm",
                "shape_yx": [2, 2],
                "command_revision": 1,
                "mapping_revision": 0,
                "receipt": {
                    "identity": "broken-simulated-slm",
                    "outcome": "known-new",
                    "command_revision": True,
                    "mapping_revision": 0.0,
                },
                "phase_bytes": len(phase),
            },
        }, phase

    monkeypatch.setattr(device_module, "_rpc_call", invalid_state)
    with pytest.raises(ValueError, match="receipt revision"):
        _RemoteSlmAdapter("127.0.0.1", 1, 1.0)

    def noncanonical_phase(endpoint, method, arguments, timeout):
        metadata, phase = invalid_state(endpoint, method, arguments, timeout)
        metadata["state"]["receipt"]["command_revision"] = 1
        metadata["state"]["receipt"]["mapping_revision"] = 0
        return metadata, phase

    monkeypatch.setattr(device_module, "_rpc_call", noncanonical_phase)
    with pytest.raises(ValueError, match="canonical snapshot"):
        _RemoteSlmAdapter("127.0.0.1", 1, 1.0)

    malformed = (
        b'{"version":1,"version":1}',
        b'{"value":NaN}',
        b"[]",
    )
    for metadata in malformed:
        sender, receiver = socket.socketpair()
        try:
            sender.sendall(device_module._REMOTE_HEADER.pack(len(metadata), 0) + metadata)
            sender.shutdown(socket.SHUT_WR)
            with pytest.raises((TypeError, ValueError)):
                device_module._recv_packet(receiver)
        finally:
            sender.close()
            receiver.close()

    sender, receiver = socket.socketpair()
    try:
        sender.sendall(device_module._REMOTE_HEADER.pack(8, 0)[:3])
        sender.close()
        with pytest.raises(ConnectionError, match="mid-message"):
            device_module._recv_packet(receiver)
    finally:
        receiver.close()

    sender, receiver = socket.socketpair()
    try:
        sender.sendall(
            device_module._REMOTE_HEADER.pack(
                0, device_module._MAX_REMOTE_PHASE_BYTES + 1
            )
        )
        with pytest.raises(ValueError, match="maximum size"):
            device_module._recv_packet(receiver)
    finally:
        sender.close()
        receiver.close()


def test_slm_server_launcher_uses_the_product_entry() -> None:
    root = Path(__file__).resolve().parents[3]
    launcher = (root / "bin" / "slm_server.bat").read_text(encoding="utf-8")
    assert 'set "ZLC_COMMAND=slm_server"' in launcher
    assert 'call "%~dp0_launch.bat" %*' in launcher

    from zou_lab_control import entry_specs

    assert entry_specs("zou_lab_control.commands")["slm_server"] == (
        "zlc_atom.devices.slm.device_types:main"
    )


def test_slm_server_prints_copyable_same_machine_and_lan_device_addresses(
    monkeypatch, capsys
) -> None:
    import zlc_atom.devices.slm.device_types as module

    monkeypatch.setattr(
        module,
        "_local_ipv4_addresses",
        lambda: ("192.168.0.20", "10.0.0.5"),
    )
    _print_client_endpoints("0.0.0.0", 18862)
    output = capsys.readouterr().out
    assert "SLM LISTEN BIND 0.0.0.0:18862" in output
    assert "same computer: host=127.0.0.1 port=18862" in output
    assert "another computer: host=192.168.0.20 port=18862" in output
    assert "another computer: host=10.0.0.5 port=18862" in output
    assert "0.0.0.0 is listen-only" in output


def test_slm_server_check_uses_the_windows_loader_without_a_pair_preflight(
    monkeypatch, capsys
) -> None:
    import zlc_atom.devices.slm.device_types as module

    calls: list[Path | None] = []
    monkeypatch.setattr(module, "_find_sdk_directory", lambda _authored="": None)
    monkeypatch.setattr(
        module,
        "_load_sdk",
        lambda directory: (calls.append(directory) or object(), None),
    )

    assert module.main(
        ["--check-config", "--transport", "usb", "--host", "127.0.0.1"]
    ) == 0
    assert calls == [None]
    assert "USB SDK=Windows loader" in capsys.readouterr().out


def test_slm_server_check_defaults_to_dvi_without_loading_the_sdk(
    monkeypatch, capsys
) -> None:
    import zlc_atom.devices.slm.device_types as module

    endpoint = {
        "name": r"\\.\DISPLAY2",
        "attached": True,
        "primary": False,
        "width": 1280,
        "height": 1024,
        "frequency": 60,
        "x": 1920,
        "y": 0,
    }
    monkeypatch.setattr(module, "_windows_displays", lambda: (endpoint,))
    monkeypatch.setattr(
        module,
        "_load_sdk",
        lambda *_args: (_ for _ in ()).throw(AssertionError("DVI loaded the SDK")),
    )

    assert module.main(["--check-config", "--host", "127.0.0.1"]) == 0
    assert r"DVI display=\\.\DISPLAY2" in capsys.readouterr().out


def test_slm_server_cli_validates_before_hardware_and_closes_after_bind_failure(
    monkeypatch,
) -> None:
    import zlc_atom.devices.slm.device_types as module

    profile_calls = 0
    original_profile = module._load_profile

    def counted_profile(name):
        nonlocal profile_calls
        profile_calls += 1
        return original_profile(name)

    monkeypatch.setattr(module, "_load_profile", counted_profile)
    for arguments in (
        ["--check-config", "--host", "bad host"],
        ["--check-config", "--port", "-1"],
    ):
        with pytest.raises(SystemExit):
            module.main(arguments)
    assert profile_calls == 0

    class FakeAdapter:
        closed = 0

        def close(self):
            self.closed += 1

    adapter = FakeAdapter()
    monkeypatch.setattr(module, "X15213Adapter", lambda _authored: adapter)
    monkeypatch.setattr(
        module,
        "_open_slm_server",
        lambda *_args: (_ for _ in ()).throw(OSError("bind failed")),
    )
    with pytest.raises(OSError, match="bind failed"):
        module.main(["--host", "127.0.0.1", "--port", "18862"])
    assert adapter.closed == 1
