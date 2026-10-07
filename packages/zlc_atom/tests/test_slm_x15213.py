from __future__ import annotations

import json
from pathlib import Path
import socket
import sys
import threading
import time
from threading import Barrier, Thread

import numpy as np
import pytest
from PIL import Image

from zlc_atom.devices.slm import SlmAdapter
from zlc_atom.devices.slm.hamamatsu_x15213.remote import _RemoteSlmAdapter, _open_slm_server
from zlc_atom.devices.slm.hamamatsu_x15213.device_types import (
    DEVICE_TYPES,
    HAMAMATSU_X15213_SCHEMA,
    X15213_LOCAL_SCHEMA,
    X15213_SERVER_SCHEMA,
    X15213Adapter,
    _load_sdk,
    _load_correction,
    _load_profile,
    _print_client_endpoints,
)
from zlc_atom.install import create_installation

from fakes import running_slm_server


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
        "device_profile": "LSH0804382",
        "wavelength_nm": 852.0,
        "correction_path": "",
        "flip_x": False,
        "flip_y": False,
    }
    values.update(changes)
    return values


def _patch_usb(monkeypatch, sdk: _UsbSdk, handle: _Handle | None = None) -> _Handle:
    import zlc_atom.devices.slm.hamamatsu_x15213.device_types as module

    result = handle or _Handle()
    monkeypatch.setattr(module, "_load_sdk", lambda: (sdk, result))
    return result


def test_real_slm_descriptor_matches_the_pulse_server_endpoint_model() -> None:
    assert [item.type_id for item in DEVICE_TYPES] == [
        "slm.hamamatsu_x15213",
        "slm.hamamatsu_x15213_local",
    ]
    descriptor, local = DEVICE_TYPES
    assert descriptor.domain == local.domain == "slm"
    assert descriptor.capabilities == local.capabilities == ("slm.phase",)
    assert descriptor.discover is None
    assert descriptor.control_factory is not None
    # The local head serves the identical protocol, so the identical
    # control surface drives it -- through its own loopback client.
    assert local.control_factory is descriptor.control_factory
    assert HAMAMATSU_X15213_SCHEMA.field_names == ("host", "port")
    assert HAMAMATSU_X15213_SCHEMA.project_values({}) == {
        "host": "127.0.0.1",
        "port": 18862,
    }
    assert set(X15213_SERVER_SCHEMA.field_names) == set(_config())
    # The local head is the server's own form plus the port to serve on --
    # the same facts the CLI takes -- so initializing it is starting the
    # server, with nothing retyped.
    assert [field.name for field in X15213_LOCAL_SCHEMA.fields] == [
        field.name for field in X15213_SERVER_SCHEMA.fields
    ] + ["port"]


def test_profile_is_strict_and_records_physical_provenance_boundaries(
    monkeypatch, tmp_path: Path
) -> None:
    import zlc_atom.devices.slm.hamamatsu_x15213.device_types as module

    profile = _load_profile("LSH0804382")
    assert profile["model"] == "X15213 (exact type suffix not recorded)"
    assert profile["serial"] == "LSH0804382"
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


def test_the_slm_server_admits_peers_only_while_told_to(monkeypatch) -> None:
    """A head a bench serves for itself answers nobody but this machine
    until the device is published; the CLI's server admits peers from the
    start."""

    sdk = _UsbSdk()
    _patch_usb(monkeypatch, sdk)
    physical = X15213Adapter(_config())
    try:
        held = _open_slm_server(physical, "127.0.0.1", 0, peers=False)
        try:
            assert held.peers is False
            assert held.verify_request(None, ("127.0.0.1", 40000)) is True
            assert held.verify_request(None, ("10.0.0.5", 40000)) is False
            held.admit_peers(True)
            assert held.verify_request(None, ("10.0.0.5", 40000)) is True
            held.admit_peers(False)
            assert held.verify_request(None, ("10.0.0.5", 40000)) is False
        finally:
            held.server_close()
        served = _open_slm_server(physical, "127.0.0.1", 0)
        try:
            assert served.peers is True
            assert served.verify_request(None, ("10.0.0.5", 40000)) is True
        finally:
            served.server_close()
    finally:
        physical.close()


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
    """Each stage of the walk -- write, display, readback, settle -- fails."""

    import zlc_atom.devices.slm.hamamatsu_x15213.device_types as module

    sdk = _UsbSdk()
    _patch_usb(monkeypatch, sdk)
    adapter = X15213Adapter(_config())
    try:
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

        sdk.check_result = 1

        # The settle is the adapter's own ``time.sleep``, and ``time`` is the
        # one module every thread in this process sleeps through: interrupt
        # only this thread's sleeps, and only for this one apply.
        settling = threading.get_ident()
        real_sleep = module.time.sleep

        def fail_settle(seconds: float) -> None:
            if threading.get_ident() != settling:
                real_sleep(seconds)
                return
            raise RuntimeError("settle interrupted")

        with monkeypatch.context() as scoped:
            scoped.setattr(module.time, "sleep", fail_settle)
            with pytest.raises(RuntimeError, match="settle interrupted"):
                adapter.apply_phase(np.zeros(adapter.shape_yx, dtype=np.float32))
        assert adapter.last_commanded_phase is None
        assert adapter.last_command_receipt["outcome"] == "unknown"
        assert adapter.last_command_receipt["stage"] == "settle"
        assert adapter.last_command_receipt["readback"] == "matched-new"
    finally:
        adapter.close()


def test_command_receipt_freezes_the_authored_correction_mapping(
    monkeypatch, tmp_path: Path
) -> None:
    """The correction is an Init field; every receipt states the mapping it used."""

    sdk = _UsbSdk()
    _patch_usb(monkeypatch, sdk)
    correction_path = tmp_path / "CAL_LSH0804382_852nm.bmp"
    correction = np.zeros((1024, 1272), dtype=np.uint8)
    correction[0, 0] = 255
    Image.fromarray(correction, mode="L").save(correction_path)

    adapter = X15213Adapter(_config(correction_path=str(correction_path)))
    try:
        adapter.apply_phase(np.zeros(adapter.shape_yx, dtype=np.float32))
        frozen = adapter.last_command_receipt
        assert frozen["mapping_revision"] == adapter.mapping_revision
        assert frozen["correction_enabled"] is True
        assert frozen["correction_path"] == str(correction_path.resolve())
        from zlc_atom.devices.slm.device import _validated_state

        with pytest.raises(ValueError, match="newer than device truth"):
            _validated_state(
                adapter.identity,
                adapter.shape_yx,
                adapter.last_commanded_phase,
                adapter.command_revision,
                adapter.mapping_revision,
                {**frozen, "mapping_revision": adapter.mapping_revision + 1},
            )
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


def test_the_sdk_is_found_through_the_vendor_folder_and_nowhere_else(
    monkeypatch, tmp_path: Path
) -> None:
    """The bench-wide vendor rule, through its one resolver.

    The loader used to walk an authored directory, ``HAMAMATSU_SLM_SDK``,
    every PATH entry, the working directory and Program Files, and then
    hand a bare file name to the Windows loader -- an SDK found "somewhere"
    that nobody could account for on the experiment machine.  Now: the
    family's ``vendor/`` folder or the absolute path its ``vendor.json``
    names, and a miss is the instruction saying so.
    """

    import json

    import zlc_atom.devices.slm.hamamatsu_x15213.device_types as module
    import zlc_atom.devices.vendor as vendor_module

    vendor = tmp_path / "vendor"
    vendor.mkdir()
    monkeypatch.setattr(vendor_module, "vendor_directory", lambda _anchor: vendor)
    with pytest.raises(FileNotFoundError) as missing:
        _load_sdk()
    assert "copy hpkSLMdaLV.dll into" in str(missing.value)
    assert str(vendor) in str(missing.value)

    elsewhere = tmp_path / "sdk" / "hpkSLMdaLV.dll"
    elsewhere.parent.mkdir()
    elsewhere.write_bytes(b"vendor library placeholder")
    (vendor / "vendor.json").write_text(
        json.dumps({"hpkSLMdaLV.dll": str(elsewhere)}), encoding="utf-8"
    )
    loaded: list[str] = []
    registered: list[str] = []
    sentinel = object()
    monkeypatch.setattr(
        module.ctypes,
        "WinDLL",
        lambda library: loaded.append(str(library)) or sentinel,
        raising=False,
    )
    monkeypatch.setattr(
        module.os,
        "add_dll_directory",
        lambda directory: registered.append(str(directory)) or _Handle(),
    )
    sdk, handle = _load_sdk()
    assert sdk is sentinel
    assert loaded == [str(elsewhere)]
    assert registered == [str(elsewhere.parent)], (
        "the SDK's sibling DLLs load from the primary library's own folder"
    )
    assert isinstance(handle, _Handle)


def test_dvi_server_transport_needs_no_vendor_dll_and_preserves_the_raster_path(
    monkeypatch,
) -> None:
    import zlc_atom.devices.slm.hamamatsu_x15213.device_types as module

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
    server, worker = running_slm_server(adapter)
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


@pytest.mark.skipif(sys.platform != "win32", reason="native DVI raster uses Windows GDI")
def test_native_dvi_raster_is_pixel_exact_and_releases_its_window(monkeypatch):
    from PIL import ImageGrab
    import zlc_atom.devices.slm.hamamatsu_x15213.device_types as module

    # Use only a normal large primary desktop. An exact SLM-sized desktop
    # must never be selected by this software-only acceptance test.
    desktop = next((item for item in module._windows_displays()
                    if item["primary"] and item["width"] > 1280 and item["height"] > 1024), None)
    if desktop is None:
        pytest.skip("pixel proof requires a normal desktop larger than the SLM raster")
    x, y = int(desktop["x"]), int(desktop["y"])
    monkeypatch.setattr(module, "_display", lambda _name: {"name": "software-test-only", "x": x, "y": y})
    yy, xx = np.ogrid[:1024, :1272]
    frame = np.asarray((37 * yy + 17 * xx) % 256, dtype=np.uint8)
    movie = (frame, np.asarray(255 - frame, dtype=np.uint8))
    present, close, prepare = module._open_dvi_presenter("software-test-only")
    try:
        prepare(movie)
        for index in (0, 1):
            present(index)
            screenshot = np.asarray(ImageGrab.grab(bbox=(x, y, x + 1280, y + 1024), all_screens=True))
            expected = np.zeros((1024, 1280), np.uint8)
            expected[:, :1272] = movie[index]
            np.testing.assert_array_equal(screenshot[:, :, :3], np.repeat(expected[:, :, None], 3, axis=2))
        prepare(())
        screenshot = np.asarray(ImageGrab.grab(bbox=(x, y, x + 1280, y + 1024), all_screens=True))
        np.testing.assert_array_equal(screenshot[:, :, :3], np.repeat(expected[:, :, None], 3, axis=2))
    finally:
        close()
    close()


def test_broken_or_missing_usb_sdk_cannot_block_the_default_dvi_transport(
    monkeypatch,
) -> None:
    import zlc_atom.devices.slm.hamamatsu_x15213.device_types as module

    for absent in (
        FileNotFoundError("the Hamamatsu SLM SDK is not installed"),
        OSError("could not load hpkSLMdaLV.dll"),
    ):
        monkeypatch.setattr(
            module,
            "_load_sdk",
            lambda error=absent: (_ for _ in ()).throw(error),
        )
        assert module._prepare_dvi_controller("LSH0804382") is False


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
    monkeypatch, tmp_path: Path,
) -> None:
    import zlc_atom.devices.slm.hamamatsu_x15213.remote as device_module

    sdk = _UsbSdk()
    handle = _patch_usb(monkeypatch, sdk)
    yy, xx = np.ogrid[:1024, :1272]
    correction = (3 * yy + 5 * xx).astype(np.uint8)
    correction_path = tmp_path / "remote-correction.bmp"
    Image.fromarray(correction).save(correction_path)
    physical = X15213Adapter(_config(
        flip_x=True, flip_y=True, correction_path=str(correction_path),
    ))
    server, worker = running_slm_server(physical)
    calls: list[str] = []
    original = device_module._rpc_call
    connections: list[socket.socket] = []
    create_connection = socket.create_connection

    def connected(*args, **kwargs):
        connection = create_connection(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(device_module.socket, "create_connection", connected)

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
        assert isinstance(remote, SlmAdapter)
        assert calls == ["describe"]
        assert remote.identity == physical.identity
        assert remote.shape_yx == physical.shape_yx
        assert remote.last_commanded_phase is None
        assert remote.command_revision == 0
        assert remote.mapping_revision == 0
        assert remote.last_command_receipt == {
            "transport": "usb",
            "identity": "hamamatsu-x15213:usb:LSH0804382",
            "profile": "LSH0804382",
            "model": "X15213 (exact type suffix not recorded)",
            "serial": "LSH0804382",
            "wavelength_nm": 852.0,
            "flip_x": True,
            "flip_y": True,
            "correction_path": str(correction_path.resolve()),
            "correction_enabled": True,
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
        assert calls == ["describe"]
        assert sdk.write_count == 0

        normalizations = []
        original_canonical = device_module.canonical_phase
        def counted_canonical(values, shape):
            normalizations.append(True)
            return original_canonical(values, shape)
        monkeypatch.setattr(device_module, 'canonical_phase', counted_canonical)
        expected = np.full(remote.shape_yx, np.pi / 3.0, dtype=np.float32)
        applied = remote.apply_phase(expected)
        assert len(normalizations) == 1, 'successful Apply normalized its owned phase again'
        assert applied is remote.last_commanded_phase
        assert calls == ["describe", "apply"]
        assert sdk.write_count == 1
        assert remote.command_revision == 1
        assert remote.last_command_receipt["outcome"] == "known-new"
        np.testing.assert_array_equal(applied, remote.last_commanded_phase)
        np.testing.assert_array_equal(applied, physical.last_commanded_phase)
        remote.apply_phase(expected)
        assert sdk.write_count == 2
        assert calls == ["describe", "apply", "apply"]
        assert len(connections) == 1

        timed_out = False

        def timeout_once(endpoint, method, arguments, timeout):
            nonlocal timed_out
            calls.append(method)
            if method == "apply_codes" and not timed_out:
                timed_out = True
                original(endpoint, method, arguments, timeout)
                raise socket.timeout("simulated reply timeout")
            return original(endpoint, method, arguments, timeout)

        monkeypatch.setattr(device_module, "_rpc_call", timeout_once)
        with pytest.raises(socket.timeout):
            remote.apply_phase_codes(
                np.full(remote.shape_yx, 64, dtype=np.uint8)
            )
        assert remote.last_commanded_phase is None
        assert remote.last_command_receipt["outcome"] == "unknown"
        assert sdk.write_count == 3, "a lost reply must not resend the applied phase"
        assert connections[0].fileno() == -1
        recovered = remote.apply_phase_codes(
            np.full(remote.shape_yx, 64, dtype=np.uint8)
        )
        assert calls == ["describe", "apply", "apply", "apply_codes", "describe", "apply_codes"]
        assert sdk.write_count == 4
        assert len(connections) == 2
        np.testing.assert_array_equal(remote.last_commanded_phase, recovered)

        codes = (17 * yy + 13 * xx).astype(np.uint8)
        def checked_codes(endpoint, method, arguments, timeout):
            if method == "apply_codes":
                assert arguments[2] == [1024, 1272]
                assert len(arguments[3]) == codes.size
                assert arguments[3] == codes.tobytes()
            return counted(endpoint, method, arguments, timeout)
        monkeypatch.setattr(device_module, "_rpc_call", checked_codes)
        decoded = remote.apply_phase_codes(codes)
        expected_phase = codes.astype(np.float32) * np.float32(2 * np.pi / 256)
        np.testing.assert_array_equal(decoded, expected_phase)
        np.testing.assert_array_equal(physical.last_commanded_phase, expected_phase)
        assert decoded is remote.last_commanded_phase
        assert not decoded.flags.writeable
        assert sdk.write_count == remote.command_revision == physical.command_revision == 5
        assert remote.last_command_receipt["readback"] == "matched-new"
        profile = _load_profile("LSH0804382")
        gray = np.floor(np.interp(
            np.arange(256) * 852.0 / (128 * profile["phase_curve_wavelength_nm"]),
            profile["phase_pi_by_gray"], np.arange(256),
        ) + 0.5).astype(np.uint8)
        mapped_codes = (codes[::-1, ::-1].astype(np.uint16) + correction) % 256
        np.testing.assert_array_equal(sdk.display, gray[mapped_codes])
        for invalid in (codes.astype(np.uint16), codes[:, :-1]):
            with pytest.raises(ValueError, match="uint8 matrix"):
                remote.apply_phase_codes(invalid)
        for payload, shape in ((codes.tobytes()[:-1], [1024, 1272]),
                               (codes.tobytes(), [1024, 1271])):
            reply, _ = original(remote._connection, "apply_codes", (
                5, 0, shape, payload,
            ), 2.0)
            assert reply["ok"] is False
            assert reply["error"] == "invalid SLM phase payload"
        assert sdk.write_count == physical.command_revision == 5
        assert calls[-1] == "apply_codes"
        remote.apply_phase(expected)
        assert sdk.write_count == remote.command_revision == 6
        assert calls[-1] == "apply"
        server.shutdown()
        server.server_close()
        worker.join(timeout=2.0)
        assert connections[1].recv(1) == b"", "server close must release idle sessions"

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
    assert all(connection.fileno() == -1 for connection in connections)
    expected_calls = ["describe", "apply", "apply", "apply_codes", "describe", "apply_codes", "apply_codes", "apply"]
    assert calls == expected_calls
    with pytest.raises(RuntimeError, match="closed"):
        remote.apply_phase(expected)
    with pytest.raises(RuntimeError, match="closed"):
        remote.apply_phase_codes(codes)
    assert calls == expected_calls


@pytest.mark.parametrize("phase_codes", (False, True))
def test_remote_slm_rejects_a_stale_writer_and_refreshes_physical_truth(
    monkeypatch, phase_codes: bool,
) -> None:
    sdk = _UsbSdk()
    _patch_usb(monkeypatch, sdk)
    physical = X15213Adapter(_config())
    server, worker = running_slm_server(physical)
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
                results[name] = (adapter.apply_phase_codes(
                    np.full(adapter.shape_yx, round(value * 128 / np.pi), dtype=np.uint8)
                ) if phase_codes else adapter.apply_phase(
                    np.full(adapter.shape_yx, value, dtype=np.float32)))
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

        phase2 = loser.apply_phase_codes(
            np.full(loser.shape_yx, 96, dtype=np.uint8)
        ) if phase_codes else loser.apply_phase(
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


@pytest.mark.parametrize("phase_codes", (False, True))
def test_remote_slm_preserves_declared_failures_and_marks_bad_replies_unknown(
    monkeypatch, phase_codes: bool,
) -> None:
    import zlc_atom.devices.slm.hamamatsu_x15213.remote as device_module

    sdk = _UsbSdk()
    _patch_usb(monkeypatch, sdk)
    physical = X15213Adapter(_config())
    server, worker = running_slm_server(physical)
    remote = None
    original = device_module._rpc_call

    def apply(value: float) -> np.ndarray:
        return (remote.apply_phase_codes(
            np.full(remote.shape_yx, round(value * 128 / np.pi), dtype=np.uint8)
        ) if phase_codes else remote.apply_phase(
            np.full(remote.shape_yx, value, dtype=np.float32)))

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
        old = apply(np.pi / 4.0)
        sdk.write_updates = False
        sdk.write_result = 0
        with pytest.raises(RuntimeError, match="Write_FMemArray"):
            apply(np.pi / 2.0)
        assert remote.last_command_receipt["outcome"] == "known-old"
        np.testing.assert_array_equal(remote.last_commanded_phase, old)

        sdk.write_updates = True
        sdk.write_result = 1

        def malformed_after_apply(endpoint, method, arguments, timeout):
            metadata, payload = original(endpoint, method, arguments, timeout)
            if method in {"apply", "apply_codes"}:
                metadata["version"] = True
            return metadata, payload

        monkeypatch.setattr(device_module, "_rpc_call", malformed_after_apply)
        with pytest.raises(ValueError, match="protocol version"):
            apply(3.0 * np.pi / 4.0)
        assert physical.command_revision == 3
        assert remote.last_commanded_phase is None
        assert remote.last_command_receipt["outcome"] == "unknown"

        monkeypatch.setattr(device_module, "_rpc_call", original)
        recovered = apply(np.pi)
        assert remote.command_revision == 4
        np.testing.assert_array_equal(remote.last_commanded_phase, recovered)

        sdk.change_result = 0
        with pytest.raises(RuntimeError, match="Change_DispSlot"):
            apply(5.0 * np.pi / 4.0)
        assert remote.command_revision == 5
        assert remote.last_commanded_phase is None
        assert remote.last_command_receipt["outcome"] == "unknown"
        sdk.change_result = 1
        sdk.bad_readback = True
        with pytest.raises(RuntimeError, match="readback differs"):
            apply(3.0 * np.pi / 2.0)
        assert sdk.write_count == remote.command_revision == 6
        assert remote.last_commanded_phase is None
        assert remote.last_command_receipt["stage"] == "readback"
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
    import zlc_atom.devices.slm.hamamatsu_x15213.remote as device_module
    # These grammar-only replies bypass the wire; the proxy still owns and
    # closes the socket supplied to that request seam.
    monkeypatch.setattr(device_module.socket, "create_connection", lambda *args, **kwargs: socket.socket())

    def huge_shape(_endpoint, _method, _arguments, _timeout):
        return {
            "version": device_module._REMOTE_VERSION,
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
            "version": device_module._REMOTE_VERSION,
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
                0, device_module._MAX_REMOTE_SEQUENCE_BYTES + 1
            )
        )
        with pytest.raises(ValueError, match="maximum size"):
            device_module._recv_packet(receiver)
    finally:
        sender.close()
        receiver.close()


def test_slm_server_command_is_the_product_entry() -> None:
    """The headless escape hatch stays installed; its .bat wrapper is gone.

    A bench normally serves its head in-process (slm.hamamatsu_x15213_local);
    this command is for a machine without a bench window.
    """

    from zou_lab_control import entry_specs

    assert entry_specs("zou_lab_control.commands")["slm_server"] == (
        "zlc_atom.devices.slm.hamamatsu_x15213.device_types:main"
    )


def test_slm_server_prints_copyable_same_machine_and_lan_device_addresses(
    monkeypatch, capsys
) -> None:
    import zlc_atom.devices.slm.hamamatsu_x15213.device_types as module

    monkeypatch.setattr(
        module,
        "local_ipv4_addresses",
        lambda: ("192.168.0.20", "10.0.0.5"),
    )
    _print_client_endpoints("0.0.0.0", 18862)
    output = capsys.readouterr().out
    assert "SLM LISTEN BIND 0.0.0.0:18862" in output
    assert "same computer: host=127.0.0.1 port=18862" in output
    assert "another computer: host=192.168.0.20 port=18862" in output
    assert "another computer: host=10.0.0.5 port=18862" in output
    assert "0.0.0.0 is listen-only" in output


def test_slm_server_check_names_the_vendor_library_it_loaded(
    monkeypatch, capsys
) -> None:
    import zlc_atom.devices.slm.hamamatsu_x15213.device_types as module

    calls: list[str] = []
    monkeypatch.setattr(module, "_sdk_library", lambda: Path("C:/vendor/hpkSLMdaLV.dll"))
    monkeypatch.setattr(
        module, "_load_sdk", lambda: (calls.append("loaded") or object(), None)
    )

    assert module.main(
        ["--check-config", "--transport", "usb", "--host", "127.0.0.1"]
    ) == 0
    assert calls == ["loaded"]
    assert "USB SDK=C:\\vendor\\hpkSLMdaLV.dll" in capsys.readouterr().out.replace(
        "/", "\\"
    )


def test_slm_server_check_defaults_to_dvi_without_loading_the_sdk(
    monkeypatch, capsys
) -> None:
    import zlc_atom.devices.slm.hamamatsu_x15213.device_types as module

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
    import zlc_atom.devices.slm.hamamatsu_x15213.device_types as module

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


def test_a_local_server_whose_thread_cannot_start_releases_what_it_opened(
    monkeypatch,
) -> None:
    """The head and the listening socket are acquired before the serving
    thread starts; a start that fails must give both back.

    ``Thread.start`` was outside every cleanup path: the head stayed open
    in the SDK and the port stayed bound, with no leaf to close either
    through -- and a server that never started cannot be ``shutdown``,
    which waits for a loop that never ran.
    """

    import socket

    import zlc_atom.devices.slm.hamamatsu_x15213.device_types as module

    sdk = _UsbSdk()
    handle = _patch_usb(monkeypatch, sdk)
    servers = []
    open_server = module._open_slm_server

    def capture(adapter, host, port, *, peers=True):
        server = open_server(adapter, host, port, peers=peers)
        servers.append(server)
        return server

    monkeypatch.setattr(module, "_open_slm_server", capture)

    class _NeverStarts(Thread):
        def start(self) -> None:
            raise RuntimeError("no thread for the SLM server")

    monkeypatch.setattr(module, "Thread", _NeverStarts)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    installation = create_installation(
        (
            {
                "key": "slm",
                "type_id": "slm.hamamatsu_x15213_local",
                "config": {**_config(), "port": port},
            },
        )
    )
    try:
        assert "no thread for the SLM server" in str(installation.failures["slm"])
        (server,) = servers
        assert server.socket.fileno() == -1, "the listening socket was released"
        assert sdk.close_count == 1, "the head was released"
        assert handle.close_count == 1
    finally:
        installation.close()


def test_remote_sequence_preloads_bulk_codes_and_paces_every_acknowledged_frame(monkeypatch, tmp_path):
    import zlc_atom.devices.slm.hamamatsu_x15213.device_types as physical_module
    import zlc_atom.devices.slm.hamamatsu_x15213.remote as remote_module

    yy, xx = np.ogrid[:1024, :1272]
    codes = np.asarray([(xx + yy + 11 * index) % 256 for index in range(16)], dtype=np.uint8)
    codes.flags.writeable = False
    correction = np.asarray((3 * yy + 5 * xx) % 256, dtype=np.uint8)
    correction_path = tmp_path / "sequence-correction.bmp"
    Image.fromarray(correction).save(correction_path)
    preload, delivered, calls, sleeps = [], [], [], []
    released = threading.Event()
    real_sleep = time.sleep

    def prepare(rasters):
        preload[:] = list(rasters)
        if not len(rasters):
            released.set()

    def present(index):
        assert type(index) is int
        raster = np.zeros((1024, 1280), np.uint8)
        raster[:, :1272] = preload[index]
        delivered.append(raster)
        real_sleep(0.026 if index == 0 else 0.002)

    monkeypatch.setattr(physical_module, "_display", lambda _name: {"name": "test-display"})
    monkeypatch.setattr(physical_module, "_prepare_dvi_controller", lambda _serial: False)
    monkeypatch.setattr(physical_module, "_open_dvi_presenter", lambda _name: (present, lambda: None, prepare))
    monkeypatch.setattr(physical_module.time, "sleep", lambda seconds: sleeps.append(seconds))
    original_rpc = remote_module._rpc_call

    def counted(endpoint, method, arguments, timeout):
        calls.append(method)
        if method == "prepare_sequence":
            assert len(arguments[-1]) == codes.nbytes > 16 * 1024 * 1024
        reply = original_rpc(endpoint, method, arguments, timeout)
        if method == "play_sequence":
            assert reply[0]["state"]["phase_bytes"] == 0 and reply[1] == b""
            assert reply[0]["sequence"]["sequence_token"] == arguments[0]
        return reply

    monkeypatch.setattr(remote_module, "_rpc_call", counted)
    physical = X15213Adapter(_config(transport="dvi", flip_x=True, flip_y=True, correction_path=str(correction_path)))
    server, worker = running_slm_server(physical)
    remote = _RemoteSlmAdapter("127.0.0.1", server.server_address[1], 3.0)
    try:
        prepared = remote.prepare_phase_sequence(codes, 1 / 60)
        assert delivered == []
        assert remote.command_revision == physical.command_revision == 0
        assert prepared["frame_count"] == len(preload) == 16
        assert "phases" not in physical._sequence
        assert physical._sequence["codes"].dtype == np.uint8
        assert all(gray.shape == physical.shape_yx for gray in preload)
        assert np.shares_memory(remote._sequence_codes, codes), "readonly solver movie is retained without another full copy"
        assert not remote._sequence_codes.flags.writeable
        result = remote.play_phase_sequence()
        assert remote._sequence_codes is None
        assert calls == ["describe", "prepare_sequence", "play_sequence"]
        assert result["played_frames"] == 16
        assert result["cancelled"] is False
        assert result["physical_vblank_observed"] is False
        assert result["acknowledgment"] == "native-gdi-flush-and-exact-raster-check"
        assert len(result["dispatch_ms"]) == len(result["acknowledged_ms"]) == 16
        assert np.all(np.asarray(result["acknowledged_ms"]) > result["dispatch_ms"])
        assert np.all(np.asarray(result["actual_frame_intervals_ms"]) >= 1000 / 60 - 0.1)
        assert result["actual_frame_intervals_ms"][0] >= 26, "late acknowledgment extends cadence without skipping a frame"
        assert sleeps == [], "sequence must not use the static apply's 50 ms sleep per frame"
        assert preload == [], "completed playback releases preloaded images"
        for frame, delivered_raster in zip(codes, delivered):
            expected = physical._phase_to_gray[(frame[::-1, ::-1].astype(np.uint16) + correction) % 256]
            np.testing.assert_array_equal(delivered_raster[:, :1272], expected)
            assert not delivered_raster[:, 1272:].any()
        np.testing.assert_array_equal(remote.last_commanded_phase, codes[-1].astype(np.float32) * np.float32(2 * np.pi / 256))
        assert remote.last_command_receipt["outcome"] == "known-new"
        assert remote.command_revision == physical.command_revision == 1
        with pytest.raises(RuntimeError, match="not been prepared"):
            remote.play_phase_sequence()
        remote.prepare_phase_sequence(codes, 1 / 60)
        released.clear()
        remote.close()
        assert released.wait(2), "a disconnected preparation owner must release unplayed frames"
        assert physical._sequence is None
        np.testing.assert_array_equal(physical.last_commanded_phase, codes[-1].astype(np.float32) * np.float32(2 * np.pi / 256))
    finally:
        remote.close()
        server.shutdown()
        server.server_close()
        worker.join(2)
        physical.close()


@pytest.mark.parametrize("ending", ["stop", "display-failure", "lost-reply", "cleanup-failure"])
def test_remote_sequence_stop_and_failures_preserve_truth_while_play_rpc_is_active(monkeypatch, ending):
    import zlc_atom.devices.slm.hamamatsu_x15213.device_types as physical_module
    import zlc_atom.devices.slm.hamamatsu_x15213.remote as remote_module

    frames, delivered = [], []
    stop_requested = threading.Event()

    def prepare(rasters):
        frames[:] = list(rasters)
        if not len(rasters) and ending in {"display-failure", "cleanup-failure"}:
            raise RuntimeError("photo release failed")

    def present(index):
        if ending == "display-failure" and index == 1:
            raise RuntimeError("frame delivery failed")
        delivered.append(index)
        if ending == "stop":
            stop_requested.set()

    monkeypatch.setattr(physical_module, "_display", lambda _name: {"name": "test-display"})
    monkeypatch.setattr(physical_module, "_prepare_dvi_controller", lambda _serial: False)
    monkeypatch.setattr(physical_module, "_open_dvi_presenter", lambda _name: (present, lambda: None, prepare))
    original_rpc = remote_module._rpc_call
    calls = []

    def rpc(endpoint, method, arguments, timeout):
        calls.append(method)
        reply = original_rpc(endpoint, method, arguments, timeout)
        if method == "play_sequence" and reply[0]["ok"]:
            assert reply[0]["state"]["phase_bytes"] == 0 and reply[1] == b""
        if ending == "lost-reply" and method == "play_sequence" and calls.count("play_sequence") == 2:
            raise socket.timeout("lost final sequence reply")
        return reply

    monkeypatch.setattr(remote_module, "_rpc_call", rpc)
    physical = X15213Adapter(_config(transport="dvi"))
    server, worker = running_slm_server(physical)
    remote = _RemoteSlmAdapter("127.0.0.1", server.server_address[1], 3.0)
    codes = np.full((3, *physical.shape_yx), np.arange(3, dtype=np.uint8)[:, None, None] * 32, dtype=np.uint8)
    try:
        if ending == "lost-reply":
            remote.prepare_phase_sequence(codes, 0.001)
            remote.play_phase_sequence()
            assert "sequence" in remote.last_command_receipt
            delivered.clear()
        remote.prepare_phase_sequence(codes, 0.2 if ending == "stop" else 0.001)
        if ending == "stop":
            assert not np.shares_memory(remote._sequence_codes, codes)
            codes[0] = 255  # Receipt reconstruction retains the actual uploaded bytes.
            result = remote.play_phase_sequence(stop_requested.is_set)
            assert result["cancelled"] is True
            assert result["played_frames"] == 1
            assert calls.count("cancel_sequence") == 1, "Stop needs an independent RPC beside active playback"
            assert remote.last_command_receipt["stage"] == "sequence-cancelled"
            np.testing.assert_array_equal(remote.last_commanded_phase, np.zeros(physical.shape_yx, np.float32))
        elif ending == "display-failure":
            with pytest.raises(RuntimeError, match="frame delivery failed"):
                remote.play_phase_sequence()
            assert remote.last_commanded_phase is None
            assert remote.last_command_receipt["outcome"] == "unknown"
            assert remote.last_command_receipt["sequence"]["played_frames"] == 1
            assert "photo release failed" in remote.last_command_receipt["sequence"]["release_error"]
        elif ending == "lost-reply":
            with pytest.raises(socket.timeout, match="lost final"):
                remote.play_phase_sequence()
            assert remote.last_commanded_phase is None
            assert remote.last_command_receipt["outcome"] == "unknown"
            assert "sequence" not in remote.last_command_receipt, "a lost reply cannot relabel the previous sequence as this run's timing"
            assert physical.last_command_receipt["sequence"]["played_frames"] == 3
        else:
            with pytest.raises(RuntimeError, match="photo release failed"):
                remote.play_phase_sequence()
            assert remote.last_command_receipt["outcome"] == "known-new"
            np.testing.assert_array_equal(remote.last_commanded_phase, codes[-1].astype(np.float32) * np.float32(2 * np.pi / 256))
        assert len(delivered) == (3 if ending in {"lost-reply", "cleanup-failure"} else 1)
        assert frames == []
        assert remote._sequence_codes is None
    finally:
        remote.close()
        server.shutdown()
        server.server_close()
        worker.join(2)
        physical.close()


def test_usb_sequence_preloads_nonvisible_slots_then_only_changes_slots_locally(monkeypatch):
    class SlotsSdk(_UsbSdk):
        def __init__(self):
            super().__init__()
            self.slots = {}
            self.changes = []
            self.fail_change_at = None

        def Write_FMemArray(self, _board, source, size, width, height, slot):
            self.write_count += 1
            self.slots[slot] = np.ctypeslib.as_array(source, shape=(int(size),)).reshape(int(height), int(width)).copy()
            return 1

        def Change_DispSlot(self, _board, slot):
            self.changes.append(slot)
            if len(self.changes) == self.fail_change_at:
                return 0
            self.display = self.slots[slot]
            return 1

    sdk = SlotsSdk()
    _patch_usb(monkeypatch, sdk)
    physical = X15213Adapter(_config())
    try:
        codes = np.asarray([np.full(physical.shape_yx, index * 37, np.uint8) for index in range(3)])
        physical.prepare_phase_sequence(codes, 0.001)
        assert sdk.write_count == 3
        assert set(sdk.slots) == {1, 2, 3}
        assert sdk.changes == []
        assert not sdk.display.any()
        result = physical.play_phase_sequence()
        assert sdk.changes == [1, 2, 3]
        assert sdk.write_count == 3, "playback must not upload each frame again"
        assert result["acknowledgment"] == "sdk-slot-change-and-frame-memory-readback"
        assert result["played_frames"] == 3
        assert physical._sequence is None
        np.testing.assert_array_equal(physical.last_commanded_phase, codes[-1].astype(np.float32) * np.float32(2 * np.pi / 256))
        physical.prepare_phase_sequence(codes, 0.001)
        assert physical._display_slot not in physical._sequence["slots"]
        physical.release_phase_sequence()
        with pytest.raises(RuntimeError, match="not been prepared"):
            physical.play_phase_sequence()
        physical.prepare_phase_sequence(codes, 0.001)
        sdk.fail_change_at = len(sdk.changes) + 2
        with pytest.raises(RuntimeError, match="Change_DispSlot"):
            physical.play_phase_sequence()
        assert physical.last_command_receipt["outcome"] == "known-old"
        confirmed = physical.last_commanded_phase
        np.testing.assert_array_equal(confirmed, codes[0].astype(np.float32) * np.float32(2 * np.pi / 256))
        with pytest.raises(ValueError):
            confirmed.flags.writeable = True
    finally:
        physical.close()


@pytest.mark.parametrize("wrong", ["token", "mapping", "revision", "played_frames"])
def test_remote_sequence_rejects_unconfirmed_metadata_instead_of_guessing_phase(monkeypatch, wrong):
    import zlc_atom.devices.slm.hamamatsu_x15213.remote as module
    from zlc_atom.devices.simulation.slm.device import VirtualSLM
    from zlc_atom.devices.simulation.world import SimulationWorld

    physical = VirtualSLM(SimulationWorld(), identity="receipt-verification")
    server, worker = running_slm_server(physical)
    original = module._rpc_call

    def altered(endpoint, method, arguments, timeout):
        metadata, payload = original(endpoint, method, arguments, timeout)
        if method == "play_sequence":
            assert metadata["ok"] and payload == b""
            if wrong == "token":
                metadata["sequence"]["sequence_token"] = "another-sequence"
            elif wrong == "mapping":
                metadata["state"]["mapping_revision"] += 1
            elif wrong == "revision":
                metadata["state"]["command_revision"] += 1
            else:
                metadata["state"]["receipt"]["sequence"]["played_frames"] = 0
        return metadata, payload

    monkeypatch.setattr(module, "_rpc_call", altered)
    remote = _RemoteSlmAdapter("127.0.0.1", server.server_address[1], 2.0)
    try:
        codes = np.full((2, *physical.shape_yx), 37, dtype=np.uint8)
        remote.prepare_phase_sequence(codes, .001)
        with pytest.raises(ValueError, match="confirmation receipt"):
            remote.play_phase_sequence()
        assert remote.last_commanded_phase is None
        assert remote.last_command_receipt["outcome"] == "unknown"
        assert remote._sequence_codes is None
        assert physical.last_command_receipt["outcome"] == "known-new"
    finally:
        remote.close()
        server.shutdown()
        server.server_close()
        worker.join(2)
        physical.close()


@pytest.mark.parametrize("ending", ["complete", "stop", "upload-failure", "lost-reply"])
def test_remote_stream_overlaps_verified_frames_and_wakes_bounded_waits(monkeypatch, ending):
    import zlc_atom.devices.slm.hamamatsu_x15213.device_types as physical_module
    import zlc_atom.devices.slm.hamamatsu_x15213.remote as remote_module

    delivered, errors, playback = [], [], []
    first_display, blocked_producer, unblock_display, producer_done = (threading.Event() for _ in range(4))
    frames = np.frombuffer(bytes(np.arange(12, dtype=np.uint8).repeat(1024 * 1272)), np.uint8).reshape(12, 1024, 1272)

    def present(gray):
        assert isinstance(gray, np.ndarray) and not gray.flags.writeable
        delivered.append(gray.copy())
        first_display.set()
        if ending == "stop":
            assert unblock_display.wait(2)

    monkeypatch.setattr(physical_module, "_display", lambda _name: {"name": "test-display"})
    monkeypatch.setattr(physical_module, "_prepare_dvi_controller", lambda _serial: False)
    monkeypatch.setattr(physical_module, "_open_dvi_presenter", lambda _name: (present, lambda: None, lambda _frames: None))
    original_rpc = remote_module._rpc_call

    def rpc(endpoint, method, arguments, timeout):
        if ending == "upload-failure" and method == "submit_sequence_frame" and arguments[1] == 2:
            raise socket.timeout("upload queue reply lost")
        reply = original_rpc(endpoint, method, arguments, timeout)
        if method == "submit_sequence_frame":
            assert reply[1] == b"" and "state" not in reply[0], "queue admission is not hardware ACK"
        if method == "play_sequence" and reply[0]["ok"]:
            assert reply[1] == b"" and reply[0]["state"]["phase_bytes"] == 0
            if ending == "lost-reply":
                raise socket.timeout("final stream reply lost")
        return reply

    monkeypatch.setattr(remote_module, "_rpc_call", rpc)
    physical = X15213Adapter(_config(transport="dvi", flip_x=True, flip_y=True))
    server, worker = running_slm_server(physical)
    remote = _RemoteSlmAdapter("127.0.0.1", server.server_address[1], 2)

    def play():
        try:
            playback.append(remote.play_phase_sequence())
        except BaseException as error:
            errors.append(error)

    def produce():
        try:
            for index, frame in enumerate(frames):
                if index == 6:
                    blocked_producer.set()
                remote.submit_phase_frame(index, frame)
                if index == 0:
                    assert first_display.wait(2), "first frame must display before the complete movie exists"
                if ending != "stop":
                    time.sleep(.023)  # Explicit simulated GPU production slower than nominal 60 Hz.
        except BaseException as error:
            errors.append(error)
        finally:
            producer_done.set()

    player = Thread(target=play)
    producer = Thread(target=produce)
    try:
        prepared = remote.prepare_phase_sequence(None, 1 / 60, frame_count=12)
        assert prepared["streaming"] and prepared["queue_capacity"] == 2
        assert physical._sequence["codes"] is None and physical._sequence["queue"].maxsize == 2
        player.start(); producer.start()
        assert first_display.wait(2) and not producer_done.is_set()
        if ending == "stop":
            assert blocked_producer.wait(2)
            assert not producer_done.wait(.04), "a full bounded stream must backpressure, never drop/overwrite"
            remote.cancel_phase_sequence()
            unblock_display.set()
        producer.join(3); player.join(3)
        assert not producer.is_alive() and not player.is_alive()
        assert remote._sequence_upload is None and remote._sequence_codes is None
        assert physical._sequence is None
        if ending == "complete":
            assert errors == [] and playback[0]["played_frames"] == len(frames)
            assert max(playback[0]["queue_wait_ms"]) > 1, "underrun waits are measured, not hidden/skipped"
            assert len(playback[0]["upload_ms"]) == len(frames)
            assert playback[0]["final_settle_completed"]
        elif ending == "stop":
            assert playback[0]["cancelled"] and playback[0]["played_frames"] == 1
            assert any("cancelled" in str(error) for error in errors)
        elif ending == "upload-failure":
            assert any("upload queue reply lost" in str(error) for error in errors)
            assert physical.last_command_receipt["sequence"]["cancelled"]
            assert remote.last_command_receipt["outcome"] == "known-new", "final play receipt confirms the partial phase"
        else:
            assert any("final stream reply lost" in str(error) for error in errors)
            assert remote.last_command_receipt["outcome"] == "unknown" and remote.last_commanded_phase is None
        for frame, gray in zip(frames, delivered):
            np.testing.assert_array_equal(gray, physical._phase_to_gray[frame[::-1, ::-1]])
        if ending != "lost-reply":
            confirmed = physical.last_command_receipt["sequence"]["played_frames"]
            np.testing.assert_array_equal(remote.last_commanded_phase,
                                          frames[confirmed - 1].astype(np.float32) * np.float32(2 * np.pi / 256))
    finally:
        unblock_display.set()
        remote.cancel_phase_sequence()
        if producer.is_alive(): producer.join(3)
        if player.is_alive(): player.join(3)
        remote.close()
        server.shutdown(); server.server_close(); worker.join(2)
        physical.close()


@pytest.mark.parametrize("admitted", [0, 1, 3])
def test_remote_stream_upload_eof_stops_only_an_unfinished_current_prefix(monkeypatch, admitted):
    import zlc_atom.devices.slm.hamamatsu_x15213.device_types as physical_module
    import zlc_atom.devices.slm.hamamatsu_x15213.remote as remote_module

    monkeypatch.setattr(physical_module, "_display", lambda _name: {"name": "test-display"})
    monkeypatch.setattr(physical_module, "_prepare_dvi_controller", lambda _serial: False)
    monkeypatch.setattr(physical_module, "_open_dvi_presenter", lambda _name: (lambda _frame: None, lambda: None, lambda _frames: None))
    physical = X15213Adapter(_config(transport="dvi"))
    initial = np.full(physical.shape_yx, np.float32(37 * 2 * np.pi / 256))
    physical.apply_phase(initial)
    server, worker = running_slm_server(physical)
    endpoint = ("127.0.0.1", server.server_address[1])
    primary = socket.create_connection(endpoint, timeout=2)
    uploader = socket.create_connection(endpoint, timeout=2)
    replies, errors = [], []
    player = None
    newcomer = None
    try:
        def prepare():
            metadata, payload = remote_module._rpc_call(primary, "prepare_sequence", (
                physical.command_revision, 0, list(physical.shape_yx), 3, [1 / 60] * 3, b"", True,
            ), 2)
            assert metadata["ok"] and payload == b""
            return metadata["sequence"]["sequence_token"]

        token = prepare()
        assert remote_module._rpc_call(uploader, "bind_sequence_upload", (token,), 2)[0]["ok"]
        if admitted == 0:
            # The old channel dies after new physical buffers exist but
            # before their new token is published. It must not cancel them.
            original_prepare = physical.prepare_phase_sequence

            def replace_stream(*args, **kwargs):
                prepared = original_prepare(*args, **kwargs)
                uploader.shutdown(socket.SHUT_RDWR)
                uploader.close()
                time.sleep(.03)
                assert not physical._sequence_cancel.is_set()
                return prepared

            with monkeypatch.context() as patch:
                patch.setattr(physical, "prepare_phase_sequence", replace_stream)
                replacement = prepare()
            assert replacement != token
            token = replacement
            uploader = socket.create_connection(endpoint, timeout=2)
            assert remote_module._rpc_call(uploader, "bind_sequence_upload", (token,), 2)[0]["ok"]

        def play():
            try:
                replies.append(remote_module._rpc_call(primary, "play_sequence", (token,), 2))
            except BaseException as error:
                errors.append(error)

        player = Thread(target=play)
        player.start()
        deadline = time.monotonic() + 2
        while not physical._sequence["playing"] and time.monotonic() < deadline:
            time.sleep(.001)
        assert physical._sequence["playing"]
        for index in range(admitted):
            frame = np.full(physical.shape_yx, index + 11, np.uint8)
            assert remote_module._rpc_call(uploader, "submit_sequence_frame", (token, index, frame), 2)[0]["ok"]
        uploader.shutdown(socket.SHUT_RDWR)
        uploader.close()
        player.join(2)
        assert not player.is_alive() and errors == []
        receipt = replies[0][0]["state"]["receipt"]["sequence"]
        assert receipt["cancelled"] is (admitted < 3)
        assert receipt["played_frames"] <= admitted
        if admitted == 3:
            assert receipt["played_frames"] == 3 and receipt["final_settle_completed"]
        # EOF released the actual command owner; a new ordinary handshake
        # must work and return the retained confirmed phase, not queued pixels.
        newcomer = _RemoteSlmAdapter(*endpoint, 2)
        confirmed = receipt["played_frames"]
        expected = initial if confirmed == 0 else np.full(physical.shape_yx, confirmed + 10, np.float32) * np.float32(2 * np.pi / 256)
        np.testing.assert_array_equal(newcomer.last_commanded_phase, expected)
        assert newcomer.last_command_receipt["outcome"] == "known-new"
        assert physical._sequence is None
    finally:
        physical.cancel_phase_sequence()
        uploader.close()
        primary.close()
        if player is not None: player.join(2)
        if newcomer is not None: newcomer.close()
        server.shutdown(); server.server_close(); worker.join(2)
        physical.close()
