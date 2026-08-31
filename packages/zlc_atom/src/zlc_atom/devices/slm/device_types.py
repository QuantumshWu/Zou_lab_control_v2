"""The network-installed X15213 leaf and its server-owned USB adapter."""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from dataclasses import replace
import json
import logging
import os
from pathlib import Path
from queue import Empty, Queue
import re
import socket
from threading import Event, Lock, Thread
import time
from typing import Callable, Mapping

import numpy as np
from PIL import Image

from zlc_atom.authoring import AuthoringChoice, AuthoringField, AuthoringSchema
from zlc_atom.install.descriptors import DeviceTypeDescriptor, InstalledLeaf

from . import open_slm_control
from .device import _RemoteSlmAdapter, _open_slm_server, bind_slm, canonical_phase


_SHAPE_YX = (1024, 1272)
_RASTER_YX = (1024, 1280)
_TWO_PI = 2.0 * np.pi
_TYPE_ID = "slm.hamamatsu_x15213"
_PROFILE_FORMAT = "zlc.slm.hamamatsu_x15213.device_profile"
_PROFILE_DIRECTORY = Path(__file__).resolve().parent / "profiles"
_SDK_LIBRARY = "hpkSLMdaLV.dll"
_REMOTE_TIMEOUT_SECONDS = 10.0


class _DisplayDeviceW(ctypes.Structure):
    _fields_ = (
        ("cb", wintypes.DWORD),
        ("DeviceName", wintypes.WCHAR * 32),
        ("DeviceString", wintypes.WCHAR * 128),
        ("StateFlags", wintypes.DWORD),
        ("DeviceID", wintypes.WCHAR * 128),
        ("DeviceKey", wintypes.WCHAR * 128),
    )


class _DevModeW(ctypes.Structure):
    """Complete Unicode DEVMODE through dmPanningHeight."""

    _fields_ = (
        ("dmDeviceName", wintypes.WCHAR * 32),
        ("dmSpecVersion", wintypes.WORD),
        ("dmDriverVersion", wintypes.WORD),
        ("dmSize", wintypes.WORD),
        ("dmDriverExtra", wintypes.WORD),
        ("dmFields", wintypes.DWORD),
        ("dmPositionX", wintypes.LONG),
        ("dmPositionY", wintypes.LONG),
        ("dmDisplayOrientation", wintypes.DWORD),
        ("dmDisplayFixedOutput", wintypes.DWORD),
        ("dmColor", ctypes.c_short),
        ("dmDuplex", ctypes.c_short),
        ("dmYResolution", ctypes.c_short),
        ("dmTTOption", ctypes.c_short),
        ("dmCollate", ctypes.c_short),
        ("dmFormName", wintypes.WCHAR * 32),
        ("dmLogPixels", wintypes.WORD),
        ("dmBitsPerPel", wintypes.DWORD),
        ("dmPelsWidth", wintypes.DWORD),
        ("dmPelsHeight", wintypes.DWORD),
        ("dmDisplayFlags", wintypes.DWORD),
        ("dmDisplayFrequency", wintypes.DWORD),
        ("dmICMMethod", wintypes.DWORD),
        ("dmICMIntent", wintypes.DWORD),
        ("dmMediaType", wintypes.DWORD),
        ("dmDitherType", wintypes.DWORD),
        ("dmReserved1", wintypes.DWORD),
        ("dmReserved2", wintypes.DWORD),
        ("dmPanningWidth", wintypes.DWORD),
        ("dmPanningHeight", wintypes.DWORD),
    )


HAMAMATSU_X15213_SCHEMA = AuthoringSchema(
    (
        AuthoringField("host", "str", "SLM server host", "127.0.0.1", required=True),
        AuthoringField(
            "port", "int", "SLM server port", 18862, minimum=1, maximum=65535
        ),
    )
)


X15213_SERVER_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "transport",
            "choice",
            "Server transport",
            "dvi",
            choices=(
                AuthoringChoice("dvi", "DVI display"),
                AuthoringChoice("usb", "USB frame memory"),
            ),
        ),
        AuthoringField("display_name", "str", "DVI display name", ""),
        AuthoringField(
            "sdk_directory",
            "folder",
            "Hamamatsu SDK directory",
            "",
        ),
        AuthoringField(
            "device_profile",
            "str",
            "Device profile",
            "LSH0804382",
        ),
        AuthoringField(
            "wavelength_nm",
            "float",
            "Wavelength (nm)",
            852.0,
            minimum=1.0,
        ),
        AuthoringField("correction_path", "str", "Vendor correction BMP", ""),
        AuthoringField("flip_x", "bool", "Flip X", False),
        AuthoringField("flip_y", "bool", "Flip Y", False),
    ),
)


def _windows_displays() -> tuple[dict[str, object], ...]:
    """Return current Windows display endpoints without guessing an EDID."""

    user32 = getattr(getattr(ctypes, "windll", None), "user32", None)
    if user32 is None:
        return ()
    found: list[dict[str, object]] = []
    index = 0
    while True:
        device = _DisplayDeviceW()
        device.cb = ctypes.sizeof(device)
        if not user32.EnumDisplayDevicesW(None, index, ctypes.byref(device), 0):
            break
        index += 1
        mode = _DevModeW()
        mode.dmSize = ctypes.sizeof(mode)
        if not user32.EnumDisplaySettingsW(device.DeviceName, -1, ctypes.byref(mode)):
            continue
        found.append(
            {
                "name": str(device.DeviceName),
                "attached": bool(device.StateFlags & 0x1),
                "primary": bool(device.StateFlags & 0x4),
                "width": int(mode.dmPelsWidth),
                "height": int(mode.dmPelsHeight),
                "frequency": int(mode.dmDisplayFrequency),
                "x": int(mode.dmPositionX),
                "y": int(mode.dmPositionY),
            }
        )
    return tuple(found)


def _eligible_dvi_displays() -> tuple[dict[str, object], ...]:
    return tuple(
        item
        for item in _windows_displays()
        if bool(item["attached"])
        and (int(item["width"]), int(item["height"])) == _RASTER_YX[::-1]
        and 55 <= int(item["frequency"]) <= 65
    )


def _display(name: str) -> dict[str, object]:
    """Resolve an explicit display or the sole eligible non-primary display."""

    requested = str(name).strip()
    eligible = _eligible_dvi_displays()
    if requested:
        for item in eligible:
            if str(item["name"]).casefold() == requested.casefold():
                return item
        raise RuntimeError(
            f"{requested!r} is not an attached 1280 x 1024 display at approximately 60 Hz"
        )
    non_primary = tuple(item for item in eligible if not bool(item.get("primary")))
    candidates = non_primary or eligible
    if len(candidates) == 1:
        return candidates[0]
    names = ", ".join(str(item["name"]) for item in candidates) or "none"
    raise RuntimeError(
        "X15213 DVI auto-selection needs exactly one eligible 1280 x 1024 "
        f"display; found {names}. Pass --display-name explicitly."
    )


def _set_dvi_thread_dpi_awareness() -> None:
    user32 = getattr(getattr(ctypes, "windll", None), "user32", None)
    setter = getattr(user32, "SetThreadDpiAwarenessContext", None)
    if setter is None:
        raise RuntimeError(
            "X15213 DVI requires per-monitor-v2 DPI awareness for an unscaled raster"
        )
    try:
        setter.argtypes = (ctypes.c_void_p,)
        setter.restype = ctypes.c_void_p
    except AttributeError:
        pass
    if not setter(ctypes.c_void_p(-4)):
        raise ctypes.WinError(ctypes.get_last_error())


def _native_dvi_client_geometry(hwnd: int) -> tuple[int, int, int, int]:
    user32 = getattr(getattr(ctypes, "windll", None), "user32", None)
    if user32 is None:
        raise RuntimeError("X15213 DVI native display checks require Windows")
    rect = wintypes.RECT()
    native_hwnd = wintypes.HWND(hwnd)
    if not user32.GetClientRect(native_hwnd, ctypes.byref(rect)):
        raise ctypes.WinError(ctypes.get_last_error())
    origin = wintypes.POINT(rect.left, rect.top)
    if not user32.ClientToScreen(native_hwnd, ctypes.byref(origin)):
        raise ctypes.WinError(ctypes.get_last_error())
    return (
        int(origin.x),
        int(origin.y),
        int(rect.right - rect.left),
        int(rect.bottom - rect.top),
    )


def _open_dvi_presenter(
    display_name: str,
) -> tuple[Callable[[np.ndarray], None], Callable[[], None]]:
    """Restore the exact physical-raster presenter used before USB-only M6."""

    geometry = _display(display_name)
    commands: Queue[object] = Queue()
    ready = Event()
    startup: list[BaseException] = []

    def run() -> None:
        try:
            _set_dvi_thread_dpi_awareness()
            import tkinter as tk
            from PIL import ImageTk

            root = tk.Tk(className="ZLC-X15213-DVI")
            root.withdraw()
            root.overrideredirect(True)
            root.configure(background="black", cursor="none")
            root.geometry(
                f"1280x1024{int(geometry['x']):+d}{int(geometry['y']):+d}"
            )
            root.attributes("-topmost", True)
            label = tk.Label(
                root, borderwidth=0, highlightthickness=0, background="black"
            )
            label.pack(fill="both", expand=True)

            def poll() -> None:
                try:
                    command = commands.get_nowait()
                except Empty:
                    root.after(2, poll)
                    return
                if command is None:
                    root.destroy()
                    return
                frame, done, result = command
                try:
                    photo = ImageTk.PhotoImage(
                        Image.fromarray(frame, mode="L"), master=root
                    )
                    label.configure(image=photo)
                    label.image = photo
                    root.update_idletasks()
                    root.update()
                    logical = (
                        root.winfo_width(),
                        root.winfo_height(),
                        label.winfo_width(),
                        label.winfo_height(),
                    )
                    native = _native_dvi_client_geometry(root.winfo_id())
                    expected = (
                        int(geometry["x"]),
                        int(geometry["y"]),
                        _RASTER_YX[1],
                        _RASTER_YX[0],
                    )
                    if logical != (1280, 1024, 1280, 1024) or native != expected:
                        raise RuntimeError(
                            "X15213 DVI presenter was scaled instead of producing "
                            "an exact 1280 x 1024 physical raster "
                            f"(logical={logical!r}, native={native!r}, expected={expected!r})"
                        )
                except BaseException as error:
                    result.append(error)
                finally:
                    done.set()
                root.after(0, poll)

            root.deiconify()
            root.lift()
            ready.set()
            root.after(0, poll)
            root.mainloop()
        except BaseException as error:
            startup.append(error)
            ready.set()

    thread = Thread(target=run, name="x15213-dvi-presenter", daemon=True)
    thread.start()
    if not ready.wait(5.0):
        raise TimeoutError("X15213 DVI presenter did not start within 5 seconds")
    if startup:
        raise RuntimeError("X15213 DVI presenter failed to start") from startup[0]

    def present(frame: np.ndarray) -> None:
        done = Event()
        result: list[BaseException] = []
        commands.put((np.array(frame, copy=True), done, result))
        if not done.wait(5.0):
            raise TimeoutError("X15213 DVI transport did not acknowledge the frame")
        if result:
            raise RuntimeError("X15213 DVI transport rejected the frame") from result[0]

    def close() -> None:
        commands.put(None)
        thread.join(5.0)
        if thread.is_alive():
            raise TimeoutError("X15213 DVI presenter did not close within 5 seconds")

    return present, close


def _find_sdk_directory(authored: str = "") -> Path | None:
    """Locate the primary SDK DLL without inventing a second dependency check."""

    direct = [
        authored,
        # The family's own vendor folder outranks every ambient location:
        # it is the ONE documented place the operator is told to use.
        str(Path(__file__).resolve().parent / "vendor"),
        os.environ.get("HAMAMATSU_SLM_SDK", ""),
        *os.environ.get("PATH", "").split(os.pathsep),
        str(Path.cwd()),
        str(Path(__file__).resolve().parent),
    ]
    seen: set[Path] = set()
    for text in direct:
        if text:
            path = Path(text).expanduser()
            path = path.parent if path.is_file() else path
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            if (resolved / _SDK_LIBRARY).is_file():
                return resolved
    for variable in ("ProgramFiles", "ProgramFiles(x86)"):
        root_text = os.environ.get(variable, "")
        if not root_text:
            continue
        root = Path(root_text)
        for pattern in ("*Hamamatsu*", "*LCOS*", "*SLM*"):
            for vendor in root.glob(pattern):
                resolved = vendor.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                for dll in vendor.rglob(_SDK_LIBRARY):
                    return dll.parent.resolve()
    return None


def _load_sdk(directory: Path | None):
    """Load through an explicit SDK folder or the normal Windows DLL search."""

    if not hasattr(ctypes, "WinDLL"):
        raise OSError("Hamamatsu X15213 USB control requires Windows")
    handle = (
        os.add_dll_directory(str(directory))
        if directory is not None and hasattr(os, "add_dll_directory")
        else None
    )
    library = str(directory / _SDK_LIBRARY) if directory is not None else _SDK_LIBRARY
    try:
        # The repository and this development machine contain no official SDK
        # header. Do not manufacture ctypes signatures: experiment-machine
        # acceptance must bind them from the installed vendor header first.
        return ctypes.WinDLL(library), handle
    except BaseException as error:
        if handle is not None:
            handle.close()
        source = str(directory) if directory is not None else "the Windows DLL search path"
        raise OSError(
            f"could not load {_SDK_LIBRARY} from {source}: "
            f"{type(error).__name__}: {error}"
        ) from error


def _local_ipv4_addresses() -> tuple[str, ...]:
    """Return unique non-loopback IPv4 addresses clients can actually use."""

    addresses: list[str] = []

    def add(value: object) -> None:
        address = str(value).strip()
        try:
            socket.inet_aton(address)
        except OSError:
            return
        if address == "0.0.0.0" or address.startswith("127.") or address in addresses:
            return
        addresses.append(address)

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 80))
            add(probe.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(
            socket.gethostname(), None, socket.AF_INET, socket.SOCK_STREAM
        ):
            add(info[4][0])
    except OSError:
        pass
    return tuple(addresses)


def _print_client_endpoints(bind_host: str, port: int) -> None:
    """Print the same-machine and LAN values to enter in the SLM device form."""

    host = str(bind_host).strip()
    port = int(port)
    wildcard = host == "0.0.0.0"
    same_host = "127.0.0.1" if wildcard or host.lower() == "localhost" else host
    print(f"SLM LISTEN BIND {host}:{port}", flush=True)
    print(
        f"SLM DEVICE ADDRESS same computer: host={same_host} port={port}",
        flush=True,
    )
    lan_addresses = (
        _local_ipv4_addresses()
        if wildcard
        else (() if same_host.startswith("127.") else (same_host,))
    )
    if wildcard:
        print("SLM NOTE 0.0.0.0 is listen-only; do not enter it as the device host", flush=True)
    if lan_addresses:
        for address in lan_addresses:
            print(
                f"SLM DEVICE ADDRESS another computer: host={address} port={port}",
                flush=True,
            )
    elif wildcard:
        print(
            "SLM DEVICE ADDRESS another computer: NOT DISCOVERED; "
            "run ipconfig and use this computer's LAN IPv4",
            flush=True,
        )


def _check(result: object, operation: str) -> None:
    if int(result) != 1:
        raise RuntimeError(f"Hamamatsu SDK {operation} failed (result={result!r})")


def _usb_open(sdk) -> int:
    ids = (ctypes.c_uint8 * 1)()
    count = int(sdk.Open_Dev(ids, 1))
    if count < 1:
        raise RuntimeError("no Hamamatsu USB SLM is connected")
    return int(ids[0])


def _usb_close(sdk, board_id: int) -> None:
    ids = (ctypes.c_uint8 * 1)(board_id)
    _check(sdk.Close_Dev(ids, 1), "Close_Dev")


def _usb_serial(sdk, board_id: int) -> str:
    value = ctypes.create_string_buffer(11)
    _check(sdk.Check_HeadSerial(board_id, value, len(value)), "Check_HeadSerial")
    return value.value.decode("ascii").strip("\x00 ")


def _usb_mode(sdk, board_id: int) -> int:
    value = ctypes.c_uint32()
    _check(sdk.Mode_Check(board_id, ctypes.byref(value)), "Mode_Check")
    return int(value.value)


def _connect_usb(sdk, serial: str) -> tuple[int, str]:
    board_id = _usb_open(sdk)
    keep_open = False
    try:
        observed = _usb_serial(sdk, board_id)
        if observed != serial:
            raise RuntimeError(
                f"X15213 profile serial {serial!r} differs from connected head {observed!r}"
            )
        mode = _usb_mode(sdk, board_id)
        if mode == 1:
            keep_open = True
            return board_id, observed
        if mode != 0:
            raise RuntimeError(f"Hamamatsu X15213 reported unknown controller mode {mode}")
        _check(sdk.Mode_Select(board_id, 1), "Mode_Select(USB)")
        _check(sdk.Reboot(board_id), "Reboot")
    finally:
        if not keep_open:
            _usb_close(sdk, board_id)

    deadline = time.monotonic() + 15.0
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        time.sleep(0.25)
        reopened: int | None = None
        keep_reopened = False
        try:
            reopened = _usb_open(sdk)
            observed = _usb_serial(sdk, reopened)
            if observed != serial:
                raise RuntimeError(
                    f"X15213 profile serial {serial!r} differs from connected head "
                    f"{observed!r} after reboot"
                )
            mode = _usb_mode(sdk, reopened)
            if mode == 1:
                keep_reopened = True
                return reopened, observed
            if mode != 0:
                raise RuntimeError(
                    f"Hamamatsu X15213 reported unknown controller mode {mode} after reboot"
                )
        except BaseException as error:
            last_error = error
        finally:
            if reopened is not None and not keep_reopened:
                _usb_close(sdk, reopened)
    raise TimeoutError(
        "X15213 rebooted for USB mode but did not reconnect within 15 seconds"
    ) from last_error


def _prepare_dvi_controller(sdk_directory: str, serial: str) -> bool:
    """Switch a reachable controller to DVI; DVI itself does not require the SDK."""

    directory = _find_sdk_directory(sdk_directory)
    if directory is None:
        return False
    try:
        sdk, handle = _load_sdk(directory)
    except OSError:
        return False
    board_id: int | None = None
    try:
        try:
            board_id = _usb_open(sdk)
        except RuntimeError:
            return False
        observed = _usb_serial(sdk, board_id)
        if observed != serial:
            raise RuntimeError(
                f"X15213 profile serial {serial!r} differs from connected head {observed!r}"
            )
        mode = _usb_mode(sdk, board_id)
        if mode == 0:
            return True
        if mode != 1:
            raise RuntimeError(f"Hamamatsu X15213 reported unknown controller mode {mode}")
        _check(sdk.Mode_Select(board_id, 0), "Mode_Select(DVI)")
        _check(sdk.Reboot(board_id), "Reboot")
        board_id = None  # Reboot has already invalidated the USB session.
        return True
    finally:
        try:
            if board_id is not None:
                _usb_close(sdk, board_id)
        finally:
            if handle is not None:
                handle.close()


_PROFILE_FIELDS = frozenset(
    {
        "format",
        "model",
        "serial",
        "default_wavelength_nm",
        "phase_curve_wavelength_nm",
        "phase_curve_source",
        "settle_seconds",
        "settle_source",
        "phase_pi_by_gray",
    }
)


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate X15213 profile field {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite X15213 profile value {value!r}")


def _load_profile(profile_name: str) -> dict[str, object]:
    requested = profile_name.strip()
    if not requested or Path(requested).name != requested:
        raise ValueError("X15213 device profile must be a local profile name")
    filename = requested if requested.casefold().endswith(".json") else f"{requested}.json"
    path = _PROFILE_DIRECTORY / filename
    if not path.is_file():
        raise FileNotFoundError(f"X15213 device profile {requested!r} was not found")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"X15213 device profile {requested!r} is not strict JSON") from error
    if not isinstance(payload, dict) or set(payload) != _PROFILE_FIELDS:
        raise ValueError(f"X15213 device profile {requested!r} has an invalid field set")
    if payload["format"] != _PROFILE_FORMAT:
        raise ValueError(f"X15213 device profile {requested!r} has an unsupported format")
    for field in ("model", "serial", "phase_curve_source", "settle_source"):
        value = payload[field]
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ValueError(f"X15213 device profile {requested!r} has invalid {field}")
    if not str(payload["model"]).startswith("X15213"):
        raise ValueError(f"X15213 device profile {requested!r} names the wrong model")
    for field in (
        "default_wavelength_nm",
        "phase_curve_wavelength_nm",
        "settle_seconds",
    ):
        raw = payload[field]
        if type(raw) not in (int, float):
            raise ValueError(f"X15213 device profile {requested!r} has invalid {field}")
        value = float(raw)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"X15213 device profile {requested!r} has invalid {field}")
        payload[field] = value
    raw_curve = payload["phase_pi_by_gray"]
    if (
        not isinstance(raw_curve, list)
        or len(raw_curve) != 256
        or any(type(value) not in (int, float) for value in raw_curve)
    ):
        raise ValueError(
            f"X15213 device profile {requested!r} needs 256 numeric phase values"
        )
    curve = np.asarray(raw_curve, dtype=np.float64)
    if (
        not np.all(np.isfinite(curve))
        or curve[0] != 0.0
        or np.any(np.diff(curve) <= 0.0)
    ):
        raise ValueError(
            f"X15213 device profile {requested!r} needs 256 strictly increasing phase values"
        )
    payload["phase_pi_by_gray"] = curve
    payload["profile_name"] = Path(filename).stem
    return payload


def _half_up(values: object) -> np.ndarray:
    return np.floor(np.asarray(values, dtype=np.float64) + 0.5)


def _phase_lut(
    curve: np.ndarray,
    wavelength_nm: float,
    phase_curve_wavelength_nm: float,
) -> tuple[np.ndarray, int]:
    phase_codes = np.arange(256, dtype=np.float64)
    target_pi = phase_codes * wavelength_nm / (128.0 * phase_curve_wavelength_nm)
    two_pi_target = 2.0 * wavelength_nm / phase_curve_wavelength_nm
    if target_pi[0] < curve[0] or two_pi_target > curve[-1]:
        raise ValueError(
            f"wavelength {wavelength_nm:g} nm lies outside the device profile phase-curve range"
        )
    gray_float = np.interp(target_pi, curve, phase_codes)
    two_pi_float = float(np.interp(two_pi_target, curve, phase_codes))
    gray = _half_up(gray_float)
    two_pi_gray = int(np.floor(two_pi_float + 0.5))
    if np.any(gray < 0.0) or np.any(gray > 255.0) or not 0 <= two_pi_gray <= 255:
        raise ValueError(
            f"wavelength {wavelength_nm:g} nm lies outside the device profile gray range"
        )
    return gray.astype(np.uint8), two_pi_gray


_WAVELENGTH_IN_NAME = re.compile(r"(?P<wavelength>\d+(?:\.\d+)?)\s*nm", re.IGNORECASE)
_CALIBRATION_NAME = re.compile(
    r"^CAL_(?P<serial>.+)_(?P<wavelength>\d+(?:\.\d+)?)nm$",
    re.IGNORECASE,
)


def _load_correction(
    path_text: str,
    *,
    expected_serial: str,
    wavelength_nm: float,
) -> np.ndarray:
    if not path_text:
        values = np.zeros(_SHAPE_YX, dtype=np.uint8)
    else:
        path = Path(path_text)
        with Image.open(path) as image:
            if image.mode != "L":
                raise ValueError("X15213 correction BMP must be an 8-bit grayscale image")
            values = np.array(image, dtype=np.uint8, copy=True)
        if values.shape != _SHAPE_YX:
            raise ValueError("X15213 correction BMP must be 1272 x 1024 pixels")
        calibration = _CALIBRATION_NAME.fullmatch(path.stem)
        if calibration is not None:
            observed_serial = calibration.group("serial")
            if observed_serial.casefold() != expected_serial.casefold():
                raise ValueError(
                    "X15213 correction calibration serial "
                    f"{observed_serial!r} does not match profile serial {expected_serial!r}"
                )
            source_wavelength = float(calibration.group("wavelength"))
        else:
            matches = tuple(_WAVELENGTH_IN_NAME.finditer(path.stem))
            source_wavelength = float(matches[-1].group("wavelength")) if matches else None
        if source_wavelength is not None and source_wavelength != wavelength_nm:
            raise ValueError(
                "X15213 correction wavelength conversion requires measured two-dimensional "
                "phase-unwrapping evidence; load a map for the configured wavelength"
            )
    return np.frombuffer(np.ascontiguousarray(values).tobytes(), dtype=np.uint8).reshape(
        _SHAPE_YX
    )


class X15213Adapter:
    """One server-owned X15213 using the proven DVI path or explicit USB."""

    shape_yx = _SHAPE_YX

    def __init__(self, values: Mapping[str, object]) -> None:
        authored = X15213_SERVER_SCHEMA.project_values(values)
        profile = _load_profile(str(authored["device_profile"]))
        self._profile_name = str(profile["profile_name"])
        self._model = str(profile["model"])
        self._profile_serial = str(profile["serial"])
        self._phase_curve_source = str(profile["phase_curve_source"])
        self._settle_source = str(profile["settle_source"])
        self._settle = float(profile["settle_seconds"])
        self._wavelength_nm = float(authored["wavelength_nm"])
        if not np.isfinite(self._wavelength_nm) or self._wavelength_nm <= 0.0:
            raise ValueError("X15213 wavelength must be finite and positive")
        self._phase_to_gray, self._two_pi_gray = _phase_lut(
            np.asarray(profile["phase_pi_by_gray"]),
            self._wavelength_nm,
            float(profile["phase_curve_wavelength_nm"]),
        )
        self._flip_x = bool(authored["flip_x"])
        self._flip_y = bool(authored["flip_y"])
        correction_text = str(authored["correction_path"])
        correction_path = str(Path(correction_text).resolve()) if correction_text else ""
        correction = _load_correction(
            correction_path,
            expected_serial=self._profile_serial,
            wavelength_nm=self._wavelength_nm,
        )

        self._state_lock = Lock()
        self._correction = correction
        self._correction_path = correction_path
        self._correction_enabled = bool(correction_path)
        self._mapping_revision = 0
        self._command_revision = 0
        self._phase: np.ndarray | None = None
        self._last_gray: np.ndarray | None = None
        self._transport = str(authored["transport"])
        self._presenter: tuple[
            Callable[[np.ndarray], None], Callable[[], None]
        ] | None = None
        self._display_name = ""
        self._dvi_controller_mode_proven = False
        self._sdk = None
        self._dll_handle = None
        self._board_id: int | None = None
        self._closed = False

        if self._transport == "dvi":
            geometry = _display(str(authored["display_name"]))
            self._display_name = str(geometry["name"]).strip()
            self._dvi_controller_mode_proven = _prepare_dvi_controller(
                str(authored["sdk_directory"]), self._profile_serial
            )
            self.identity = f"hamamatsu-x15213:dvi-display:{self._display_name}"
        else:
            directory = _find_sdk_directory(str(authored["sdk_directory"]))
            self._sdk, self._dll_handle = _load_sdk(directory)
            try:
                self._board_id, serial = _connect_usb(self._sdk, self._profile_serial)
            except BaseException:
                if self._dll_handle is not None:
                    self._dll_handle.close()
                raise
            self.identity = f"hamamatsu-x15213:usb:{serial}"
        self._last_receipt = self._receipt(
            self._mapping_snapshot(),
            outcome="unknown",
            stage="uncommanded",
            readback="not-run",
        )

    def _mapping_snapshot(self) -> dict[str, object]:
        with self._state_lock:
            return {
                "correction": self._correction,
                "correction_path": self._correction_path,
                "correction_enabled": self._correction_enabled,
                "mapping_revision": self._mapping_revision,
            }

    def _receipt(
        self,
        mapping: Mapping[str, object],
        *,
        outcome: str,
        stage: str,
        readback: str,
    ) -> dict[str, object]:
        return {
            "transport": self._transport,
            "identity": self.identity,
            "profile": self._profile_name,
            "model": self._model,
            "serial": self._profile_serial,
            "wavelength_nm": self._wavelength_nm,
            "flip_x": self._flip_x,
            "flip_y": self._flip_y,
            "correction_path": str(mapping["correction_path"]),
            "correction_enabled": bool(mapping["correction_enabled"]),
            "mapping_revision": int(mapping["mapping_revision"]),
            "settle_seconds": self._settle,
            "settle_source": self._settle_source,
            "phase_curve_source": self._phase_curve_source,
            "dvi_controller_mode_proven": self._dvi_controller_mode_proven,
            "outcome": outcome,
            "command_revision": self._command_revision,
            "stage": stage,
            "readback": readback,
        }

    @property
    def last_commanded_phase(self) -> np.ndarray | None:
        with self._state_lock:
            return self._phase

    @property
    def command_revision(self) -> int:
        with self._state_lock:
            return self._command_revision

    @property
    def mapping_revision(self) -> int:
        with self._state_lock:
            return self._mapping_revision

    @property
    def last_command_receipt(self) -> dict[str, object]:
        with self._state_lock:
            return dict(self._last_receipt)

    @property
    def wavelength_nm(self) -> float:
        return self._wavelength_nm

    @property
    def two_pi_gray(self) -> float:
        return float(self._two_pi_gray)

    @property
    def correction_name(self) -> str:
        with self._state_lock:
            return Path(self._correction_path).name if self._correction_path else ""

    @property
    def correction_enabled(self) -> bool:
        with self._state_lock:
            return self._correction_enabled

    def load_correction(self, path: str | Path) -> int:
        path_text = str(path)
        if not path_text:
            raise ValueError("X15213 correction load requires a local BMP path")
        resolved = str(Path(path_text).resolve())
        correction = _load_correction(
            resolved,
            expected_serial=self._profile_serial,
            wavelength_nm=self._wavelength_nm,
        )
        with self._state_lock:
            self._correction = correction
            self._correction_path = resolved
            self._correction_enabled = True
            self._mapping_revision += 1
            return self._mapping_revision

    def set_correction_enabled(self, enabled: bool) -> int:
        requested = bool(enabled)
        with self._state_lock:
            if requested and not self._correction_path:
                raise RuntimeError("no X15213 correction map is loaded")
            if requested != self._correction_enabled:
                self._correction_enabled = requested
                self._mapping_revision += 1
            return self._mapping_revision

    def _gray(self, canonical: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
        mapping = self._mapping_snapshot()
        oriented = canonical
        if self._flip_y:
            oriented = oriented[::-1, :]
        if self._flip_x:
            oriented = oriented[:, ::-1]
        phase_code = np.mod(
            _half_up(oriented.astype(np.float64) * (128.0 / np.pi)), 256.0
        ).astype(np.uint16)
        if bool(mapping["correction_enabled"]):
            phase_code = (
                phase_code + np.asarray(mapping["correction"], dtype=np.uint16)
            ) % 256
        return (
            np.ascontiguousarray(self._phase_to_gray[phase_code], dtype=np.uint8),
            mapping,
        )

    def _readback(self) -> np.ndarray:
        observed = np.empty(_SHAPE_YX, dtype=np.uint8)
        target = observed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        _check(
            self._sdk.Check_Disp_IMG(
                self._board_id,
                int(observed.size),
                _SHAPE_YX[1],
                _SHAPE_YX[0],
                target,
            ),
            "Check_Disp_IMG",
        )
        return observed

    def _record_failure(
        self,
        *,
        stage: str,
        gray: np.ndarray,
        mapping: Mapping[str, object],
        previous_phase: np.ndarray | None,
        previous_gray: np.ndarray | None,
        previous_receipt: Mapping[str, object],
        observed: np.ndarray | None = None,
    ) -> None:
        readback = "failed"
        if stage != "settle" and observed is None:
            try:
                observed = self._readback()
            except BaseException:
                observed = None
        if observed is not None:
            if np.array_equal(observed, gray):
                readback = "matched-new"
            elif previous_gray is not None and np.array_equal(observed, previous_gray):
                readback = "matched-old"
            else:
                readback = "mismatch"

        outcome = "unknown"
        known_phase: np.ndarray | None = None
        known_gray: np.ndarray | None = None
        receipt_mapping = mapping
        if readback == "matched-old" and previous_phase is not None:
            outcome = "known-old"
            known_phase = previous_phase
            known_gray = previous_gray
            receipt_mapping = {
                "correction_path": previous_receipt["correction_path"],
                "correction_enabled": previous_receipt["correction_enabled"],
                "mapping_revision": previous_receipt["mapping_revision"],
            }

        with self._state_lock:
            self._phase = known_phase
            self._last_gray = known_gray
            self._last_receipt = self._receipt(
                receipt_mapping,
                outcome=outcome,
                stage=stage,
                readback=readback,
            )

    def _record_unknown(
        self,
        *,
        stage: str,
        mapping: Mapping[str, object],
        readback: str,
    ) -> None:
        with self._state_lock:
            self._phase = None
            self._last_gray = None
            self._last_receipt = self._receipt(
                mapping,
                outcome="unknown",
                stage=stage,
                readback=readback,
            )

    def apply_phase(self, radians: object) -> np.ndarray:
        if self._closed:
            raise RuntimeError("X15213 is closed")
        canonical = canonical_phase(radians, _SHAPE_YX)
        gray, mapping = self._gray(canonical)
        with self._state_lock:
            previous_phase = self._phase
            previous_gray = self._last_gray
            previous_receipt = self._last_receipt
            self._command_revision += 1
            self._phase = None
            self._last_gray = None
            self._last_receipt = self._receipt(
                mapping,
                outcome="unknown",
                stage="write-pending",
                readback="not-run",
            )

        if self._transport == "dvi":
            try:
                if self._presenter is None:
                    self._presenter = _open_dvi_presenter(self._display_name)
                raster = np.zeros(_RASTER_YX, dtype=np.uint8)
                raster[:, : _SHAPE_YX[1]] = gray
                self._presenter[0](raster)
            except BaseException:
                self._record_unknown(
                    stage="display",
                    mapping=mapping,
                    readback="not-available",
                )
                raise
            try:
                if self._settle:
                    time.sleep(self._settle)
            except BaseException:
                self._record_unknown(
                    stage="settle",
                    mapping=mapping,
                    readback="not-available",
                )
                raise
            with self._state_lock:
                self._phase = canonical
                self._last_gray = gray
                self._last_receipt = self._receipt(
                    mapping,
                    outcome="known-new",
                    stage="complete",
                    readback="presenter-ack",
                )
            return canonical

        source = gray.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        try:
            _check(
                self._sdk.Write_FMemArray(
                    self._board_id,
                    source,
                    int(gray.size),
                    _SHAPE_YX[1],
                    _SHAPE_YX[0],
                    0,
                ),
                "Write_FMemArray",
            )
        except BaseException:
            self._record_failure(
                stage="write",
                gray=gray,
                mapping=mapping,
                previous_phase=previous_phase,
                previous_gray=previous_gray,
                previous_receipt=previous_receipt,
            )
            raise
        try:
            _check(self._sdk.Change_DispSlot(self._board_id, 0), "Change_DispSlot")
        except BaseException:
            self._record_failure(
                stage="display",
                gray=gray,
                mapping=mapping,
                previous_phase=previous_phase,
                previous_gray=previous_gray,
                previous_receipt=previous_receipt,
            )
            raise
        try:
            observed = self._readback()
        except BaseException:
            self._record_failure(
                stage="readback",
                gray=gray,
                mapping=mapping,
                previous_phase=previous_phase,
                previous_gray=previous_gray,
                previous_receipt=previous_receipt,
            )
            raise
        if not np.array_equal(observed, gray):
            self._record_failure(
                stage="readback",
                gray=gray,
                mapping=mapping,
                previous_phase=previous_phase,
                previous_gray=previous_gray,
                previous_receipt=previous_receipt,
                observed=observed,
            )
            raise RuntimeError("X15213 USB frame-memory readback differs from the command")
        try:
            if self._settle:
                time.sleep(self._settle)
        except BaseException:
            self._record_failure(
                stage="settle",
                gray=gray,
                mapping=mapping,
                previous_phase=previous_phase,
                previous_gray=previous_gray,
                previous_receipt=previous_receipt,
                observed=observed,
            )
            raise
        with self._state_lock:
            self._phase = canonical
            self._last_gray = gray
            self._last_receipt = self._receipt(
                mapping,
                outcome="known-new",
                stage="complete",
                readback="matched-new",
            )
        return canonical

    def close(self) -> None:
        if self._closed:
            return
        if self._transport == "dvi":
            if self._presenter is not None:
                self._presenter[1]()
                self._presenter = None
            self._closed = True
            return
        if self._board_id is not None:
            _usb_close(self._sdk, self._board_id)
            self._board_id = None
        if self._dll_handle is not None:
            self._dll_handle.close()
            self._dll_handle = None
        self._sdk = None
        self._closed = True


def _factory(context, key: str, values: Mapping[str, object]) -> InstalledLeaf:
    authored = HAMAMATSU_X15213_SCHEMA.project_values(values)
    adapter = _RemoteSlmAdapter(
        str(authored["host"]),
        int(authored["port"]),
        _REMOTE_TIMEOUT_SECONDS,
    )
    return bind_slm(context, key, adapter, _TYPE_ID)


_LOG = logging.getLogger(__name__)

#: The machine the SLM head is plugged into serves it FROM the bench process:
#: initialize the device and the server is up -- no .bat, no second console.
#: The bench's own leaf dials the same loopback endpoint every remote client
#: would, so the hardware adapter has exactly one owner: the server.
X15213_LOCAL_SCHEMA = AuthoringSchema(
    X15213_SERVER_SCHEMA.fields
    + (
        AuthoringField(
            "port", "int", "Serve on port", 18862, minimum=1, maximum=65535
        ),
    ),
)


def _local_factory(context, key: str, values: Mapping[str, object]) -> InstalledLeaf:
    """Open the plugged-in head, serve it, and join as the loopback client."""

    authored = X15213_LOCAL_SCHEMA.project_values(values)
    adapter = X15213Adapter(
        X15213_SERVER_SCHEMA.project_values(
            {field.name: authored[field.name] for field in X15213_SERVER_SCHEMA.fields}
        )
    )
    try:
        server = _open_slm_server(adapter, "0.0.0.0", int(authored["port"]))
    except BaseException:
        adapter.close()
        raise
    port = int(server.server_address[1])
    thread = Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.1},
        name=f"slm-server:{port}",
        daemon=True,
    )
    thread.start()

    def _stop_server() -> None:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)
        adapter.close()
        _LOG.info("SLM SERVER CLOSED device=%s", adapter.identity)

    try:
        client = _RemoteSlmAdapter("127.0.0.1", port, _REMOTE_TIMEOUT_SECONDS)
        leaf = bind_slm(context, key, client, "slm.hamamatsu_x15213_local")
    except BaseException:
        _stop_server()
        raise

    def _close(client=client) -> None:
        try:
            client.close()
        finally:
            _stop_server()

    return replace(leaf, closer=_close)


def _announce_local(parameters: Mapping[str, object]) -> tuple[str, dict]:
    """A peer reaches this head as an ordinary network SLM client."""

    return _TYPE_ID, {"host": "127.0.0.1", "port": int(parameters["port"])}


DEVICE_TYPES = (
    DeviceTypeDescriptor(
        _TYPE_ID,
        "slm",
        HAMAMATSU_X15213_SCHEMA,
        ("slm.phase",),
        factory=_factory,
        control_factory=open_slm_control,
    ),
    DeviceTypeDescriptor(
        "slm.hamamatsu_x15213_local",
        "slm",
        X15213_LOCAL_SCHEMA,
        ("slm.phase",),
        factory=_local_factory,
        control_factory=open_slm_control,
        announce=_announce_local,
        log_channels=("zlc_atom.devices.slm",),
    ),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Serve one local Hamamatsu SLM to local or LAN clients."
    )
    parser.add_argument("--host", default=os.environ.get("ZLC_SLM_HOST", "0.0.0.0"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("ZLC_SLM_PORT", "18862"))
    )
    parser.add_argument(
        "--transport",
        choices=("dvi", "usb"),
        default=os.environ.get("ZLC_SLM_TRANSPORT", "dvi"),
    )
    parser.add_argument(
        "--display-name", default=os.environ.get("ZLC_SLM_DISPLAY", "")
    )
    parser.add_argument("--sdk-directory", default="")
    parser.add_argument("--device-profile", default="LSH0804382")
    parser.add_argument("--wavelength-nm", type=float, default=852.0)
    parser.add_argument("--correction-path", default="")
    parser.add_argument("--flip-x", action="store_true")
    parser.add_argument("--flip-y", action="store_true")
    parser.add_argument("--check-config", action="store_true")
    arguments = parser.parse_args(argv)
    if (
        not arguments.host.strip()
        or arguments.host != arguments.host.strip()
        or any(character.isspace() for character in arguments.host)
    ):
        parser.error("SLM server host must be non-empty text without whitespace")
    if not 1 <= arguments.port <= 65535:
        parser.error("SLM server port must be from 1 through 65535")
    authored = X15213_SERVER_SCHEMA.project_values(
        {
            "transport": arguments.transport,
            "display_name": arguments.display_name,
            "sdk_directory": arguments.sdk_directory,
            "device_profile": arguments.device_profile,
            "wavelength_nm": arguments.wavelength_nm,
            "correction_path": arguments.correction_path,
            "flip_x": arguments.flip_x,
            "flip_y": arguments.flip_y,
        }
    )
    profile = _load_profile(str(authored["device_profile"]))
    _phase_lut(
        np.asarray(profile["phase_pi_by_gray"], dtype=np.float64),
        float(authored["wavelength_nm"]),
        float(profile["phase_curve_wavelength_nm"]),
    )
    if authored["correction_path"]:
        _load_correction(
            str(authored["correction_path"]),
            expected_serial=str(profile["serial"]),
            wavelength_nm=float(authored["wavelength_nm"]),
        )
    if arguments.check_config:
        if authored["transport"] == "dvi":
            display = _display(str(authored["display_name"]))
            detail = f"DVI display={display['name']}"
        else:
            directory = _find_sdk_directory(str(authored["sdk_directory"]))
            sdk, dll_handle = _load_sdk(directory)
            del sdk
            if dll_handle is not None:
                dll_handle.close()
            detail = f"USB SDK={directory if directory is not None else 'Windows loader'}"
        print(
            "SLM server config OK: "
            f"{profile['serial']} at {arguments.host}:{arguments.port}; {detail}"
        )
        return 0

    # Headless CLI: the same narration an embedding bench window shows goes
    # to this console, so the two ways of serving read identically.
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
    logger = logging.getLogger("zlc_atom.devices.slm")
    logger.addHandler(console)
    if logger.getEffectiveLevel() > logging.INFO:
        logger.setLevel(logging.INFO)

    adapter = X15213Adapter(authored)
    try:
        with _open_slm_server(adapter, arguments.host, arguments.port) as server:
            address = server.server_address
            print(
                f"ZLC SLM SERVER READY {address[0]}:{address[1]} "
                f"device={adapter.identity}",
                flush=True,
            )
            _print_client_endpoints(arguments.host, int(address[1]))
            try:
                server.serve_forever(poll_interval=0.1)
            except KeyboardInterrupt:
                pass
    finally:
        adapter.close()
    return 0


__all__ = [
    "DEVICE_TYPES",
    "HAMAMATSU_X15213_SCHEMA",
    "X15213_LOCAL_SCHEMA",
    "X15213_SERVER_SCHEMA",
    "X15213Adapter",
]
