"""Hamamatsu X15213 hardware leaf for USB frame memory or DVI display."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
from queue import Empty, Queue
import re
from threading import Event, Lock, Thread
import time
from typing import Callable, Mapping

import numpy as np
from PIL import Image

from zlc_atom.authoring import AuthoringChoice, AuthoringField, AuthoringSchema
from zlc_atom.install.configuration import DeviceInstanceConfig
from zlc_atom.install.descriptors import DeviceTypeDescriptor, InstalledLeaf

from . import open_slm_control
from .device import bind_slm, canonical_phase


_SHAPE_YX = (1024, 1272)
_RASTER_YX = (1024, 1280)
_TWO_PI = 2.0 * np.pi
_TYPE_ID = "slm.hamamatsu_x15213"
_PROFILE_FORMAT = "zlc.slm.hamamatsu_x15213.device_profile"
_PROFILE_VERSION = 1
_PROFILE_DIRECTORY = Path(__file__).resolve().parent / "profiles"


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
    """Complete Unicode DEVMODE through dmPanningHeight (220 bytes on Windows)."""

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
        AuthoringField(
            "transport",
            "choice",
            "Transport",
            "dvi",
            choices=(
                AuthoringChoice("dvi", "DVI display"),
                AuthoringChoice("usb", "USB frame memory"),
            ),
        ),
        AuthoringField(
            "display_name",
            "str",
            "DVI display name",
            "",
            enabled_when=("transport", ("dvi",)),
        ),
        AuthoringField(
            "sdk_directory",
            "folder",
            "Hamamatsu SDK directory",
            "",
            enabled_when=("transport", ("usb",)),
        ),
        AuthoringField(
            "serial",
            "str",
            "Head serial",
            "",
            enabled_when=("transport", ("usb",)),
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
        AuthoringField("settle_seconds", "float", "Optical settle (s)", 0.05, minimum=0.0),
    ),
)


def _windows_displays() -> tuple[dict[str, object], ...]:
    """Return current Windows display endpoints; no EDID identity is inferred."""

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
                "width": int(mode.dmPelsWidth),
                "height": int(mode.dmPelsHeight),
                "frequency": int(mode.dmDisplayFrequency),
                "x": int(mode.dmPositionX),
                "y": int(mode.dmPositionY),
            }
        )
    return tuple(found)


def _display(name: str) -> dict[str, object]:
    requested = name.strip()
    for item in _windows_displays():
        if str(item["name"]).casefold() == requested.casefold():
            if not item["attached"] or (item["width"], item["height"]) != _RASTER_YX[::-1]:
                break
            if not 55 <= int(item["frequency"]) <= 65:
                break
            return item
    raise RuntimeError(
        f"{requested!r} is not an attached 1280 x 1024 display at approximately 60 Hz"
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
            label = tk.Label(root, borderwidth=0, highlightthickness=0, background="black")
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
                    photo = ImageTk.PhotoImage(Image.fromarray(frame, mode="L"), master=root)
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
                    expected_native = (
                        int(geometry["x"]),
                        int(geometry["y"]),
                        _RASTER_YX[1],
                        _RASTER_YX[0],
                    )
                    if logical != (1280, 1024, 1280, 1024) or native != expected_native:
                        raise RuntimeError(
                            "X15213 DVI presenter was scaled instead of producing "
                            "an exact 1280 x 1024 physical raster on the selected "
                            f"display (logical={logical!r}, native={native!r}, "
                            f"expected={expected_native!r})"
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
    direct = (
        authored,
        os.environ.get("HAMAMATSU_SLM_SDK", ""),
        str(Path.cwd()),
        str(Path(__file__).resolve().parent),
    )
    for text in direct:
        if text:
            path = Path(text).expanduser()
            path = path.parent if path.is_file() else path
            if (path / "hpkSLMdaLV.dll").is_file() and (path / "hpkSLMda.dll").is_file():
                return path.resolve()
    for variable in ("ProgramFiles", "ProgramFiles(x86)"):
        root_text = os.environ.get(variable, "")
        if not root_text:
            continue
        root = Path(root_text)
        seen: set[Path] = set()
        for pattern in ("*Hamamatsu*", "*LCOS*", "*SLM*"):
            for vendor in root.glob(pattern):
                resolved = vendor.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                for dll in vendor.rglob("hpkSLMdaLV.dll"):
                    if (dll.parent / "hpkSLMda.dll").is_file():
                        return dll.parent.resolve()
    return None


def _load_sdk(directory: Path):
    if not hasattr(ctypes, "WinDLL"):
        raise OSError("Hamamatsu X15213 USB control requires Windows")
    handle = os.add_dll_directory(str(directory)) if hasattr(os, "add_dll_directory") else None
    try:
        return ctypes.WinDLL(str(directory / "hpkSLMdaLV.dll")), handle
    except BaseException:
        if handle is not None:
            handle.close()
        raise


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


def _close_usb_after_probe(sdk, board_id: int) -> None:
    try:
        _usb_close(sdk, board_id)
    except BaseException:
        pass


def _connect_usb(sdk, serial: str) -> tuple[int, str]:
    requested = serial.strip()
    board_id = _usb_open(sdk)
    keep_open = False
    try:
        observed = _usb_serial(sdk, board_id)
        if requested and observed != requested:
            raise RuntimeError(
                f"requested X15213 serial {requested!r}, found {observed!r}"
            )
        mode = _usb_mode(sdk, board_id)
        if mode == 1:
            keep_open = True
            return board_id, observed
        if mode != 0:
            raise RuntimeError(
                f"Hamamatsu X15213 reported unknown controller mode {mode}"
            )
        _check(sdk.Mode_Select(board_id, 1), "Mode_Select(USB)")
        sdk.Reboot(board_id)
    finally:
        if not keep_open:
            _close_usb_after_probe(sdk, board_id)
    deadline = time.monotonic() + 15.0
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        time.sleep(0.25)
        reopened: int | None = None
        keep_reopened = False
        try:
            reopened = _usb_open(sdk)
            observed = _usb_serial(sdk, reopened)
            if requested and observed != requested:
                raise RuntimeError(
                    f"requested X15213 serial {requested!r}, found {observed!r} after reboot"
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
                _close_usb_after_probe(sdk, reopened)
    raise TimeoutError(
        "X15213 rebooted for USB mode but did not reconnect within 15 seconds"
    ) from last_error


def _prepare_dvi_controller(sdk_directory: str, serial: str) -> bool:
    """Prove/switch DVI mode when USB is reachable; False means endpoint-only evidence."""

    directory = _find_sdk_directory(sdk_directory)
    if directory is None:
        return False
    sdk, handle = _load_sdk(directory)
    board_id: int | None = None
    try:
        ids = (ctypes.c_uint8 * 1)()
        if int(sdk.Open_Dev(ids, 1)) < 1:
            return False
        board_id = int(ids[0])
        observed = _usb_serial(sdk, board_id)
        requested = serial.strip()
        if requested and observed != requested:
            raise RuntimeError(
                f"requested X15213 serial {requested!r}, found {observed!r}"
            )
        mode = _usb_mode(sdk, board_id)
        if mode == 0:
            return True
        if mode != 1:
            raise RuntimeError(
                f"Hamamatsu X15213 reported unknown controller mode {mode}"
            )
        _check(sdk.Mode_Select(board_id, 0), "Mode_Select(DVI)")
        sdk.Reboot(board_id)
        return True
    finally:
        try:
            if board_id is not None:
                _close_usb_after_probe(sdk, board_id)
        finally:
            if handle is not None:
                handle.close()


_PROFILE_FIELDS = frozenset(
    {
        "format",
        "version",
        "model",
        "serial",
        "default_wavelength_nm",
        "readout_wavelength_nm",
        "phase_pi_by_gray",
    }
)
_WAVELENGTH_IN_NAME = re.compile(r"(?P<wavelength>\d+(?:\.\d+)?)\s*nm", re.IGNORECASE)
_CALIBRATION_NAME = re.compile(
    r"^CAL_(?P<serial>.+)_(?P<wavelength>\d+(?:\.\d+)?)nm$",
    re.IGNORECASE,
)


def _load_profile(profile_name: str) -> dict[str, object]:
    requested = profile_name.strip()
    if not requested or Path(requested).name != requested:
        raise ValueError("X15213 device profile must be a local profile name")
    filename = requested if requested.casefold().endswith(".json") else f"{requested}.json"
    path = _PROFILE_DIRECTORY / filename
    if not path.is_file():
        raise FileNotFoundError(f"X15213 device profile {requested!r} was not found")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"X15213 device profile {requested!r} is not valid JSON") from error
    if not isinstance(payload, dict) or set(payload) != _PROFILE_FIELDS:
        raise ValueError(f"X15213 device profile {requested!r} has an invalid field set")
    if payload["format"] != _PROFILE_FORMAT or payload["version"] != _PROFILE_VERSION:
        raise ValueError(f"X15213 device profile {requested!r} has an unsupported format")
    if payload["model"] != "X15213":
        raise ValueError(f"X15213 device profile {requested!r} names the wrong model")
    serial = str(payload["serial"]).strip()
    if not serial:
        raise ValueError(f"X15213 device profile {requested!r} has no serial")
    for field in (
        "default_wavelength_nm",
        "readout_wavelength_nm",
    ):
        value = float(payload[field])
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"X15213 device profile {requested!r} has invalid {field}")
        payload[field] = value
    curve = np.asarray(payload["phase_pi_by_gray"], dtype=np.float64)
    if (
        curve.shape != (256,)
        or not np.all(np.isfinite(curve))
        or curve[0] != 0.0
        or np.any(np.diff(curve) <= 0.0)
    ):
        raise ValueError(
            f"X15213 device profile {requested!r} needs 256 strictly increasing phase values"
        )
    payload["serial"] = serial
    payload["phase_pi_by_gray"] = curve
    payload["profile_name"] = Path(filename).stem
    return payload


def _half_up(values: object) -> np.ndarray:
    return np.floor(np.asarray(values, dtype=np.float64) + 0.5)


def _phase_lut(
    curve: np.ndarray,
    wavelength_nm: float,
    readout_wavelength_nm: float,
) -> tuple[np.ndarray, int]:
    phase_codes = np.arange(256, dtype=np.float64)
    target_pi = phase_codes * wavelength_nm / (128.0 * readout_wavelength_nm)
    two_pi_target = 2.0 * wavelength_nm / readout_wavelength_nm
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


def _correction_source_wavelength(path: Path, expected_serial: str) -> float | None:
    calibration = _CALIBRATION_NAME.fullmatch(path.stem)
    if calibration is not None:
        observed_serial = calibration.group("serial")
        if observed_serial.casefold() != expected_serial.casefold():
            raise ValueError(
                "X15213 correction calibration serial "
                f"{observed_serial!r} does not match device profile serial {expected_serial!r}"
            )
        return float(calibration.group("wavelength"))
    matches = tuple(_WAVELENGTH_IN_NAME.finditer(path.stem))
    if matches:
        return float(matches[-1].group("wavelength"))
    return None


def _scale_correction_wavelength(
    values: np.ndarray,
    source_wavelength_nm: float,
    wavelength_nm: float,
) -> np.ndarray:
    if source_wavelength_nm == wavelength_nm:
        return values
    wrapped_radians = values.astype(np.float64) * (_TWO_PI / 256.0)
    unwrapped = np.unwrap(wrapped_radians, axis=1)
    unwrapped = np.unwrap(unwrapped, axis=0)
    scaled = np.remainder(
        unwrapped * (source_wavelength_nm / wavelength_nm), _TWO_PI
    )
    return np.mod(_half_up(scaled * (256.0 / _TWO_PI)), 256.0).astype(np.uint8)


def _load_correction(
    path_text: str,
    *,
    expected_serial: str,
    wavelength_nm: float,
) -> np.ndarray:
    if not path_text:
        return np.zeros(_SHAPE_YX, dtype=np.uint8)
    path = Path(path_text)
    with Image.open(path) as image:
        if image.mode != "L":
            raise ValueError("X15213 correction BMP must be an 8-bit grayscale image")
        values = np.array(image, dtype=np.uint8, copy=True)
    if values.shape != _SHAPE_YX:
        raise ValueError("X15213 correction BMP must be 1272 x 1024 pixels")
    source_wavelength = _correction_source_wavelength(path, expected_serial)
    if source_wavelength is None:
        return values
    if not np.isfinite(source_wavelength) or source_wavelength <= 0.0:
        raise ValueError("X15213 correction BMP filename has an invalid wavelength")
    return _scale_correction_wavelength(values, source_wavelength, wavelength_nm)


class X15213Adapter:
    """One phase-only X15213; a DVI frame ack is transport evidence only."""

    shape_yx = _SHAPE_YX

    def __init__(self, values: Mapping[str, object]) -> None:
        authored = HAMAMATSU_X15213_SCHEMA.project_values(values)
        profile = _load_profile(str(authored["device_profile"]))
        self._profile_serial = str(profile["serial"])
        requested_serial = str(authored["serial"]).strip()
        if requested_serial and requested_serial != self._profile_serial:
            raise ValueError(
                f"X15213 serial {requested_serial!r} does not match device profile "
                f"serial {self._profile_serial!r}"
            )
        self._wavelength_nm = float(authored["wavelength_nm"])
        if not np.isfinite(self._wavelength_nm) or self._wavelength_nm <= 0.0:
            raise ValueError("X15213 wavelength must be finite and positive")
        self._readout_wavelength_nm = float(profile["readout_wavelength_nm"])
        self._phase_curve = np.array(profile["phase_pi_by_gray"], copy=True)
        self._phase_to_gray, self._two_pi_gray = _phase_lut(
            self._phase_curve,
            self._wavelength_nm,
            self._readout_wavelength_nm,
        )
        self._transport = str(authored["transport"])
        self._presenter = None
        self._display_name = ""
        self._dvi_controller_mode_proven = False
        self._sdk = None
        self._dll_handle = None
        self._board_id: int | None = None
        self._closed = False
        if self._transport == "dvi":
            requested_display = str(authored["display_name"]).strip()
            if not requested_display:
                raise ValueError("X15213 DVI transport requires display_name")
            geometry = _display(requested_display)
            self._display_name = str(geometry["name"]).strip()
            self._dvi_controller_mode_proven = _prepare_dvi_controller(
                str(authored["sdk_directory"]), self._profile_serial
            )
            self.identity = f"hamamatsu-x15213:dvi-display:{self._display_name}"
        self._flip_x = bool(authored["flip_x"])
        self._flip_y = bool(authored["flip_y"])
        self._settle = float(authored["settle_seconds"])
        correction_path = str(authored["correction_path"])
        self._correction_lock = Lock()
        self._correction = _load_correction(
            correction_path,
            expected_serial=self._profile_serial,
            wavelength_nm=self._wavelength_nm,
        )
        self._correction_path = correction_path
        self._correction_enabled = bool(correction_path)
        self._phase = canonical_phase(np.zeros(_SHAPE_YX), _SHAPE_YX)
        if self._transport == "usb":
            directory = _find_sdk_directory(str(authored["sdk_directory"]))
            if directory is None:
                raise FileNotFoundError(
                    "the official Hamamatsu installation's hpkSLMdaLV.dll and "
                    "hpkSLMda.dll were not found together; "
                    "set sdk_directory or HAMAMATSU_SLM_SDK"
                )
            self._sdk, self._dll_handle = _load_sdk(directory)
            try:
                self._board_id, serial = _connect_usb(self._sdk, self._profile_serial)
            except BaseException:
                if self._dll_handle is not None:
                    self._dll_handle.close()
                raise
            self.identity = f"hamamatsu-x15213:usb:{serial}"

    @property
    def last_commanded_phase(self) -> np.ndarray:
        return self._phase

    @property
    def wavelength_nm(self) -> float:
        return self._wavelength_nm

    @property
    def two_pi_gray(self) -> float:
        return float(self._two_pi_gray)

    @property
    def correction_path(self) -> str:
        with self._correction_lock:
            return self._correction_path

    @property
    def correction_name(self) -> str:
        with self._correction_lock:
            return Path(self._correction_path).name if self._correction_path else ""

    @property
    def correction_enabled(self) -> bool:
        with self._correction_lock:
            return self._correction_enabled

    @property
    def correction_available(self) -> bool:
        return True

    def load_correction(self, path: str | Path) -> None:
        path_text = str(path)
        if not path_text:
            raise ValueError("X15213 correction load requires a local BMP path")
        correction = _load_correction(
            path_text,
            expected_serial=self._profile_serial,
            wavelength_nm=self._wavelength_nm,
        )
        with self._correction_lock:
            self._correction = correction
            self._correction_path = path_text
            self._correction_enabled = True

    def set_correction_enabled(self, enabled: bool) -> None:
        requested = bool(enabled)
        with self._correction_lock:
            if requested and not self._correction_path:
                raise RuntimeError("no X15213 correction map is loaded")
            self._correction_enabled = requested

    def _gray(self, canonical: np.ndarray) -> np.ndarray:
        oriented = canonical
        if self._flip_y:
            oriented = oriented[::-1, :]
        if self._flip_x:
            oriented = oriented[:, ::-1]
        phase_code = np.mod(
            _half_up(oriented.astype(np.float64) * (128.0 / np.pi)), 256.0
        ).astype(np.uint16)
        with self._correction_lock:
            correction_enabled = self._correction_enabled
            correction_values = self._correction
        if correction_enabled:
            phase_code = (
                phase_code + correction_values.astype(np.uint16)
            ) % 256
        return np.asarray(self._phase_to_gray[phase_code], dtype=np.uint8)

    def apply_phase(self, radians: object) -> np.ndarray:
        if self._closed:
            raise RuntimeError("X15213 is closed")
        canonical = canonical_phase(radians, _SHAPE_YX)
        gray = np.ascontiguousarray(self._gray(canonical))
        if self._transport == "dvi":
            if self._presenter is None:
                self._presenter = _open_dvi_presenter(self._display_name)
            raster = np.zeros(_RASTER_YX, dtype=np.uint8)
            raster[:, : _SHAPE_YX[1]] = gray
            self._presenter[0](raster)
        else:
            size = int(gray.size)
            source = gray.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
            _check(
                self._sdk.Write_FMemArray(
                    self._board_id, source, size, _SHAPE_YX[1], _SHAPE_YX[0], 0
                ),
                "Write_FMemArray",
            )
            _check(self._sdk.Change_DispSlot(self._board_id, 0), "Change_DispSlot")
            observed = np.empty(_SHAPE_YX, dtype=np.uint8)
            target = observed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
            _check(
                self._sdk.Check_Disp_IMG(
                    self._board_id, size, _SHAPE_YX[1], _SHAPE_YX[0], target
                ),
                "Check_Disp_IMG",
            )
            if not np.array_equal(observed, gray):
                raise RuntimeError("X15213 USB frame-memory readback differs from the command")
        if self._settle:
            time.sleep(self._settle)
        self._phase = canonical
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


def _candidate_values(**changes: object) -> dict[str, object]:
    values = HAMAMATSU_X15213_SCHEMA.project_values({})
    values.update(changes)
    return values


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_") or "candidate"


def _discover_usb() -> tuple[DeviceInstanceConfig, ...]:
    directory = _find_sdk_directory()
    if directory is None:
        return ()
    sdk, handle = _load_sdk(directory)
    board_id: int | None = None
    try:
        ids = (ctypes.c_uint8 * 1)()
        if int(sdk.Open_Dev(ids, 1)) < 1:
            return ()
        board_id = int(ids[0])
        serial = _usb_serial(sdk, board_id)
        profile_name = serial
        wavelength_nm = float(
            _load_profile(profile_name)["default_wavelength_nm"]
        )
        return (
            DeviceInstanceConfig(
                f"x15213_usb_{_slug(serial)}",
                f"x15213_usb_candidate_{_slug(serial)}",
                _TYPE_ID,
                _candidate_values(
                    transport="usb",
                    display_name="",
                    sdk_directory=str(directory),
                    serial=serial,
                    device_profile=profile_name,
                    wavelength_nm=wavelength_nm,
                ),
            ),
        )
    finally:
        try:
            if board_id is not None:
                _usb_close(sdk, board_id)
        finally:
            if handle is not None:
                handle.close()


def _discover_x15213() -> tuple[DeviceInstanceConfig, ...]:
    dvi = tuple(
        DeviceInstanceConfig(
            f"x15213_dvi_{_slug(str(item['name']))}",
            f"x15213_dvi_candidate_{_slug(str(item['name']))}",
            _TYPE_ID,
            _candidate_values(display_name=str(item["name"])),
        )
        for item in _windows_displays()
        if item["attached"]
        and (item["width"], item["height"]) == _RASTER_YX[::-1]
        and 55 <= int(item["frequency"]) <= 65
    )
    return dvi + _discover_usb()


def _factory(context, key: str, values: Mapping[str, object]) -> InstalledLeaf:
    allowed = set(HAMAMATSU_X15213_SCHEMA.field_names)
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown X15213 configuration fields: {sorted(unknown)}")
    authored = {
        name: values[name]
        for name in HAMAMATSU_X15213_SCHEMA.field_names
        if name in values
    }
    adapter = X15213Adapter(authored)
    return bind_slm(context, key, adapter, _TYPE_ID)


DEVICE_TYPES = (
    DeviceTypeDescriptor(
        _TYPE_ID,
        "slm",
        HAMAMATSU_X15213_SCHEMA,
        ("slm.phase",),
        factory=_factory,
        discover=_discover_x15213,
        control_factory=open_slm_control,
    ),
)


__all__ = ["DEVICE_TYPES", "HAMAMATSU_X15213_SCHEMA", "X15213Adapter"]
