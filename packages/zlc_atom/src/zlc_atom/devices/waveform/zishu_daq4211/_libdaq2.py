"""The slice of ZishuTech's libdaq2 this driver speaks, bound with ctypes.

The vendor ships a Python wrapper of its own; this is not it.  That wrapper
loads the DLL at IMPORT time from a path inside its own package, which would
make merely listing this bench's device types fail on a machine without the
card -- and it has no argument types, so a wrong Python int reaches the
board as whatever the C compiler made of it.  This binding declares every
signature, resolves the DLL through the bench's one vendor rule, and is
opened only when a device is.

The library is a process-wide singleton addressed by device serial: there is
no handle, and two objects naming one serial are one card to it.  So the
serial IS the identity the installation's broker guards.
"""

from __future__ import annotations

import atexit
import ctypes
import threading
from typing import Any

import numpy as np

from zlc_atom.devices.vendor import resolve_vendor_file


#: What the vendor names the shared library on Windows.
LIBRARY_NAME = "daqlib2.dll"

#: Return codes this driver reads by name; everything else is reported with
#: the library's own two strings.
_SUCCESS = 0
_TIMEOUT = -7

#: How long ``libdaq2_init`` may take: it enumerates USB and then walks every
#: Ethernet adapter, which is seconds on a machine with several.
_INITIALISE_SECONDS = 20.0

_TEXT_SIZE = 256


class LibDaq2Error(RuntimeError):
    """A libdaq2 call that failed, in the library's own words."""


class LibDaq2:
    """One process's libdaq2: the calls, the strings, and the lifetime.

    Every call is serialized here.  The manual says the API is mutex
    protected, but the reads this driver makes come from a worker thread
    while the same card's properties are written from the thread that armed
    it, and one lock is cheaper to reason about than a vendor's promise.
    """

    def __init__(self, library_path: str) -> None:
        dll = ctypes.CDLL(library_path)
        self._dll = dll
        self._lock = threading.RLock()
        self._path = library_path
        text, uint = ctypes.c_char_p, ctypes.c_uint
        declare = (
            ("libdaq2_init", (), ctypes.c_int),
            ("libdaq2_exit", (), None),
            ("libdaq2_device_get_count", (), ctypes.c_int),
            ("libdaq2_device_get_sn", (uint, text, ctypes.c_int), ctypes.c_int),
            ("libdaq2_device_get_model", (text, text, ctypes.c_int), ctypes.c_int),
            ("libdaq2_device_open", (text,), ctypes.c_int),
            ("libdaq2_device_close", (text,), ctypes.c_int),
            ("libdaq2_send_command", (text, text, text), ctypes.c_int),
            ("libdaq2_set_propertyInt", (text, text, text, ctypes.c_int), ctypes.c_int),
            ("libdaq2_get_propertyInt", (text, text, text, ctypes.POINTER(ctypes.c_int)), ctypes.c_int),
            ("libdaq2_set_propertyString", (text, text, text, text), ctypes.c_int),
            ("libdaq2_adc_sync_channelsetting", (text, text), ctypes.c_int),
            ("libdaq2_adc_clear_buffer", (text, text), ctypes.c_int),
            (
                "libdaq2_adc_read_analog_sync",
                (
                    text,
                    text,
                    ctypes.POINTER(ctypes.c_double),
                    uint,
                    ctypes.POINTER(uint),
                    ctypes.c_int,
                ),
                ctypes.c_int,
            ),
        )
        for name, argtypes, restype in declare:
            function = getattr(dll, name)
            function.argtypes = list(argtypes)
            function.restype = restype
        for name in ("libdaq2_get_error_str", "libdaq2_get_error_desc"):
            function = getattr(dll, name)
            function.argtypes = [ctypes.c_int]
            function.restype = ctypes.c_char_p
        self._started = False

    @property
    def path(self) -> str:
        return self._path

    def _describe(self, code: int) -> str:
        def read(name: str) -> str:
            answer = getattr(self._dll, name)(code)
            return "" if answer is None else answer.decode("ascii", "replace")

        return f"{read('libdaq2_get_error_str')}: {read('libdaq2_get_error_desc')}"

    def _check(self, code: int, what: str) -> None:
        if code != _SUCCESS:
            raise LibDaq2Error(f"{what} failed ({code}) {self._describe(code)}")

    def start(self) -> None:
        """Initialise the library once for this process, and leave it up.

        The vendor's exit releases every card at once, so it belongs to the
        process ending and to nothing smaller: a second card being closed
        must not take this one's stream down.
        """

        with self._lock:
            if self._started:
                return
            self._check(self._dll.libdaq2_init(), "libdaq2_init")
            self._started = True
            atexit.register(self._stop)

    def _stop(self) -> None:
        with self._lock:
            if self._started:
                self._started = False
                self._dll.libdaq2_exit()

    # ------------------------------------------------------------ devices
    def device_serials(self) -> tuple[str, ...]:
        """Every card attached to this machine, by its own serial number."""

        self.start()
        with self._lock:
            count = int(self._dll.libdaq2_device_get_count())
            if count < 0:
                raise LibDaq2Error(
                    f"libdaq2_device_get_count failed ({count}) {self._describe(count)}"
                )
            serials = []
            for index in range(count):
                buffer = ctypes.create_string_buffer(_TEXT_SIZE)
                self._check(
                    self._dll.libdaq2_device_get_sn(index, buffer, _TEXT_SIZE),
                    f"reading the serial of device {index}",
                )
                serials.append(buffer.value.decode("ascii", "replace"))
            return tuple(serials)

    def device_model(self, serial: str) -> str:
        self.start()
        with self._lock:
            buffer = ctypes.create_string_buffer(_TEXT_SIZE)
            self._check(
                self._dll.libdaq2_device_get_model(
                    serial.encode("ascii"), buffer, _TEXT_SIZE
                ),
                f"reading the model of {serial}",
            )
            return buffer.value.decode("ascii", "replace")

    def open(self, serial: str) -> None:
        self.start()
        with self._lock:
            self._check(
                self._dll.libdaq2_device_open(serial.encode("ascii")),
                f"opening {serial}",
            )

    def close(self, serial: str) -> None:
        with self._lock:
            self._check(
                self._dll.libdaq2_device_close(serial.encode("ascii")),
                f"closing {serial}",
            )

    # ------------------------------------------- properties and commands
    def command(self, serial: str, module: str, command: str) -> None:
        with self._lock:
            self._check(
                self._dll.libdaq2_send_command(
                    serial.encode("ascii"), module.encode("ascii"), command.encode("ascii")
                ),
                f"{module} {command}",
            )

    def set_int(self, serial: str, module: str, name: str, value: int) -> None:
        with self._lock:
            self._check(
                self._dll.libdaq2_set_propertyInt(
                    serial.encode("ascii"),
                    module.encode("ascii"),
                    name.encode("ascii"),
                    int(value),
                ),
                f"setting {module}.{name} to {value}",
            )

    def get_int(self, serial: str, module: str, name: str) -> int:
        with self._lock:
            answer = ctypes.c_int(0)
            self._check(
                self._dll.libdaq2_get_propertyInt(
                    serial.encode("ascii"),
                    module.encode("ascii"),
                    name.encode("ascii"),
                    ctypes.byref(answer),
                ),
                f"reading {module}.{name}",
            )
            return int(answer.value)

    def set_text(self, serial: str, module: str, name: str, value: str) -> None:
        with self._lock:
            self._check(
                self._dll.libdaq2_set_propertyString(
                    serial.encode("ascii"),
                    module.encode("ascii"),
                    name.encode("ascii"),
                    value.encode("ascii"),
                ),
                f"setting {module}.{name} to {value!r}",
            )

    # --------------------------------------------------------------- ADC
    def sync_channel_setting(self, serial: str, module: str) -> None:
        with self._lock:
            self._check(
                self._dll.libdaq2_adc_sync_channelsetting(
                    serial.encode("ascii"), module.encode("ascii")
                ),
                f"{module} channel settings",
            )

    def clear_buffer(self, serial: str, module: str) -> None:
        with self._lock:
            self._check(
                self._dll.libdaq2_adc_clear_buffer(
                    serial.encode("ascii"), module.encode("ascii")
                ),
                f"clearing the {module} buffer",
            )

    def read_analog(
        self, serial: str, module: str, samples: int, timeout_ms: int
    ) -> np.ndarray:
        """Up to ``samples`` interleaved volts, or fewer when the wait expires.

        A short answer is how this library says "not yet": it warns and
        returns what it has, so the caller keeps whatever it got and asks
        again rather than treating the gap as a fault.
        """

        buffer = (ctypes.c_double * int(samples))()
        actual = ctypes.c_uint(0)
        with self._lock:
            code = self._dll.libdaq2_adc_read_analog_sync(
                serial.encode("ascii"),
                module.encode("ascii"),
                buffer,
                ctypes.c_uint(int(samples)),
                ctypes.byref(actual),
                ctypes.c_int(int(timeout_ms)),
            )
        if code not in (_SUCCESS, _TIMEOUT):
            raise LibDaq2Error(f"reading {module} failed ({code}) {self._describe(code)}")
        return np.frombuffer(buffer, dtype=np.float64, count=int(actual.value)).copy()


_LIBRARY: LibDaq2 | None = None
_LIBRARY_LOCK = threading.Lock()


def library() -> LibDaq2:
    """This process's libdaq2, loaded from the device's own vendor folder."""

    global _LIBRARY
    with _LIBRARY_LOCK:
        if _LIBRARY is None:
            _LIBRARY = LibDaq2(
                resolve_vendor_file(
                    __file__, LIBRARY_NAME, what="the ZishuTech libdaq2 SDK (64-bit)"
                )
            )
        return _LIBRARY


def set_library(value: Any) -> None:
    """Install a stand-in for the vendor library (a test bench, a fake card)."""

    global _LIBRARY
    with _LIBRARY_LOCK:
        _LIBRARY = value


__all__ = [
    "LIBRARY_NAME",
    "LibDaq2",
    "LibDaq2Error",
    "library",
    "set_library",
]
