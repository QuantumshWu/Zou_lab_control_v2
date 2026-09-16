"""Vaunix Lab Brick LMS synthesizer as an RF source, over the vendor DLL.

A Lab Brick is a USB HID device driven through Vaunix's ``vnx_fmsynth``
DLL, whose API speaks the instrument's own integer units: frequency in
10 Hz steps, power in quarter-dB steps.  The driver is written against a
small library Protocol so tests and the virtual bench stand in for the DLL
alone -- the unit conversions, the grid refusals and the read-back
discipline all run as shipped.

The grids are the honest part.  A requested value that is not exactly
representable in the instrument's units is REFUSED before it is written,
naming the step -- writing it would silently round, and the scan's dataset
column would then say something the hardware never did.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from zlc_atom.devices.rf.contract import FREQUENCY_FIELD, POWER_FIELD, RfSourceBase, snap_to_grid
from zlc_atom.devices.vendor import resolve_vendor_file

#: The instrument's own units, from the vendor API reference.
FREQUENCY_UNIT_HZ = 10.0
POWER_UNIT_DBM = 0.25


class LmsLibrary(Protocol):
    """The slice of the ``vnx_fmsynth`` API this driver consumes."""

    def device_serials(self) -> tuple[int, ...]: ...

    def open_device(self, serial: int) -> int: ...

    def close_device(self, handle: int) -> None: ...

    def set_frequency(self, handle: int, frequency_units: int) -> None: ...

    def get_frequency(self, handle: int) -> int: ...

    def set_power(self, handle: int, power_units: int) -> None: ...

    def get_power(self, handle: int) -> int: ...

    def set_rf_on(self, handle: int, enabled: bool) -> None: ...

    def get_rf_on(self, handle: int) -> bool: ...

    def get_frequency_limits(self, handle: int) -> tuple[int, int]: ...

    def get_power_limits(self, handle: int) -> tuple[int, int]: ...


class CtypesLmsLibrary:
    """The real DLL behind the Protocol.  Windows only, by the vendor."""

    @staticmethod
    def _check_status(operation: str, handle: int, status: int) -> None:
        # Vendor API manual section 3.2 and LMSTest::CheckAPISet: only bit
        # 31 denotes a command error; a nonzero success is not a refusal.
        code = int(status) & 0xFFFFFFFF
        if code & 0x80000000:
            reason = {
                0x80000000: "INVALID_DEVID",
                0x80010000: "BAD_PARAMETER",
                0x80020000: "BAD_HID_IO",
                0x80030000: "DEVICE_NOT_READY",
            }.get(code, "SDK_ERROR")
            raise RuntimeError(
                f"{operation} refused for device {handle}: {reason} "
                f"(status {int(status)}, 0x{code:08X})"
            )

    def __init__(self, dll_path: str) -> None:
        import ctypes

        if not isinstance(dll_path, str) or not dll_path.strip():
            raise ValueError("the Vaunix DLL path is required")
        self._dll = ctypes.CDLL(dll_path.strip())
        # Real hardware, not the vendor's built-in simulator.
        self._dll.fnLMS_SetTestMode(ctypes.c_bool(False))

    def device_serials(self) -> tuple[int, ...]:
        import ctypes

        count = int(self._dll.fnLMS_GetNumDevices())
        if count <= 0:
            return ()
        identifiers = (ctypes.c_uint * count)()
        self._dll.fnLMS_GetDevInfo(identifiers)
        return tuple(
            int(self._dll.fnLMS_GetSerialNumber(identifier))
            for identifier in identifiers
        )

    def open_device(self, serial: int) -> int:
        import ctypes

        count = int(self._dll.fnLMS_GetNumDevices())
        identifiers = (ctypes.c_uint * max(count, 1))()
        self._dll.fnLMS_GetDevInfo(identifiers)
        for identifier in identifiers[:count]:
            if int(self._dll.fnLMS_GetSerialNumber(identifier)) == int(serial):
                self._check_status("fnLMS_InitDevice", int(identifier),
                                   self._dll.fnLMS_InitDevice(identifier))
                return int(identifier)
        raise LookupError(f"no Vaunix LMS with serial {serial} is attached")

    def close_device(self, handle: int) -> None:
        self._check_status("fnLMS_CloseDevice", handle, self._dll.fnLMS_CloseDevice(handle))

    def set_frequency(self, handle: int, frequency_units: int) -> None:
        self._check_status("fnLMS_SetFrequency", handle,
                           self._dll.fnLMS_SetFrequency(handle, int(frequency_units)))

    def get_frequency(self, handle: int) -> int:
        return int(self._dll.fnLMS_GetFrequency(handle))

    def set_power(self, handle: int, power_units: int) -> None:
        self._check_status("fnLMS_SetPowerLevel", handle,
                           self._dll.fnLMS_SetPowerLevel(handle, int(power_units)))

    def get_power(self, handle: int) -> int:
        return int(self._dll.fnLMS_GetAbsPowerLevel(handle))

    def set_rf_on(self, handle: int, enabled: bool) -> None:
        self._check_status("fnLMS_SetRFOn", handle,
                           self._dll.fnLMS_SetRFOn(handle, bool(enabled)))

    def get_rf_on(self, handle: int) -> bool:
        return bool(self._dll.fnLMS_GetRF_On(handle))

    def get_frequency_limits(self, handle: int) -> tuple[int, int]:
        return (
            int(self._dll.fnLMS_GetMinFreq(handle)),
            int(self._dll.fnLMS_GetMaxFreq(handle)),
        )

    def get_power_limits(self, handle: int) -> tuple[int, int]:
        return (
            int(self._dll.fnLMS_GetMinPwr(handle)),
            int(self._dll.fnLMS_GetMaxPwr(handle)),
        )


@dataclass(frozen=True)
class VaunixLmsConfig:
    """Which brick, plus any optional bench-policy window.

    WHERE the vendor DLL lives is a machine fact, not an apparatus fact:
    the driver looks in this family's ``vendor/`` folder (see the README
    there), so the configuration never carries a path nobody can check.
    """

    serial: int
    frequency_low_hz: float | None = None
    frequency_high_hz: float | None = None
    power_low_dbm: float | None = None
    power_high_dbm: float | None = None


class VaunixLmsRfSource(RfSourceBase):
    def __init__(self, config: VaunixLmsConfig, *, library: LmsLibrary | None = None) -> None:
        self.config = config
        # The authored half first, with nothing open: a window that cannot
        # be honoured is refused before a USB handle exists to leak.
        super().__init__(
            frequency_low_hz=config.frequency_low_hz,
            frequency_high_hz=config.frequency_high_hz,
            power_low_dbm=config.power_low_dbm,
            power_high_dbm=config.power_high_dbm,
        )
        self._library = (
            library
            if library is not None
            else CtypesLmsLibrary(
                resolve_vendor_file(
                    __file__,
                    "vnx_fmsynth.dll",
                    what="the Vaunix LMS SDK (64-bit)",
                )
            )
        )
        self._handle = self._library.open_device(int(config.serial))
        # From here on the handle is this object's to close: a failure
        # before the constructor returns has no other owner to hand it to.
        try:
            self._attach(f"vaunix-lms:{int(config.serial)}")
        except BaseException as error:
            try:
                self._library.close_device(self._handle)
            except BaseException as close_error:
                error.add_note(
                    "closing the brick also reported: "
                    f"{type(close_error).__name__}: {close_error}"
                )
            raise

    # ------------------------------------------------------- transport verbs
    # A Lab Brick has one output, so the channel is always the bare "".
    def _write_frequency(self, channel: str, value_hz: float) -> float:
        snap_to_grid(
            value_hz, FREQUENCY_UNIT_HZ, name=FREQUENCY_FIELD, unit="Hz"
        )
        self._library.set_frequency(
            self._handle, round(value_hz / FREQUENCY_UNIT_HZ)
        )
        return self._read_frequency(channel)

    def _write_power(self, channel: str, value_dbm: float) -> float:
        snap_to_grid(value_dbm, POWER_UNIT_DBM, name=POWER_FIELD, unit="dBm")
        self._library.set_power(
            self._handle, round(value_dbm / POWER_UNIT_DBM)
        )
        return self._read_power(channel)

    def _write_output(self, channel: str, enabled: bool) -> bool:
        del channel
        self._library.set_rf_on(self._handle, enabled)
        return self._read_output("")

    def _read_frequency(self, channel: str) -> float:
        del channel
        return float(self._library.get_frequency(self._handle)) * FREQUENCY_UNIT_HZ

    def _read_power(self, channel: str) -> float:
        del channel
        return float(self._library.get_power(self._handle)) * POWER_UNIT_DBM

    def _read_output(self, channel: str) -> bool:
        del channel
        return bool(self._library.get_rf_on(self._handle))

    def _read_frequency_limits(self, channel: str) -> tuple[float, float]:
        del channel
        low, high = self._library.get_frequency_limits(self._handle)
        return float(low) * FREQUENCY_UNIT_HZ, float(high) * FREQUENCY_UNIT_HZ

    def _read_power_limits(self, channel: str) -> tuple[float, float]:
        del channel
        low, high = self._library.get_power_limits(self._handle)
        return float(low) * POWER_UNIT_DBM, float(high) * POWER_UNIT_DBM

    def close(self) -> None:
        self._library.close_device(self._handle)


__all__ = [
    "CtypesLmsLibrary",
    "FREQUENCY_UNIT_HZ",
    "LmsLibrary",
    "POWER_UNIT_DBM",
    "VaunixLmsConfig",
    "VaunixLmsRfSource",
]
