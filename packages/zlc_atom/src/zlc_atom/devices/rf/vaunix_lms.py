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

from zlc_atom.devices.rf.contract import RfSourceBase, snap_to_grid

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


class CtypesLmsLibrary:
    """The real DLL behind the Protocol.  Windows only, by the vendor."""

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
                status = int(self._dll.fnLMS_InitDevice(identifier))
                if status != 0:
                    raise RuntimeError(
                        f"Vaunix LMS {serial} refused to open (status {status})"
                    )
                return int(identifier)
        raise LookupError(f"no Vaunix LMS with serial {serial} is attached")

    def close_device(self, handle: int) -> None:
        self._dll.fnLMS_CloseDevice(handle)

    def set_frequency(self, handle: int, frequency_units: int) -> None:
        status = int(self._dll.fnLMS_SetFrequency(handle, int(frequency_units)))
        if status != 0:
            raise RuntimeError(f"fnLMS_SetFrequency refused (status {status})")

    def get_frequency(self, handle: int) -> int:
        return int(self._dll.fnLMS_GetFrequency(handle))

    def set_power(self, handle: int, power_units: int) -> None:
        status = int(self._dll.fnLMS_SetPowerLevel(handle, int(power_units)))
        if status != 0:
            raise RuntimeError(f"fnLMS_SetPowerLevel refused (status {status})")

    def get_power(self, handle: int) -> int:
        return int(self._dll.fnLMS_GetAbsPowerLevel(handle))

    def set_rf_on(self, handle: int, enabled: bool) -> None:
        self._dll.fnLMS_SetRFOn(handle, bool(enabled))

    def get_rf_on(self, handle: int) -> bool:
        return bool(self._dll.fnLMS_GetRF_On(handle))


@dataclass(frozen=True)
class VaunixLmsConfig:
    """Which brick, through which DLL, inside which authored window."""

    serial: int
    dll_path: str = "vnx_fmsynth.dll"
    frequency_low_hz: float = 500e6
    frequency_high_hz: float = 8e9
    power_low_dbm: float = -40.0
    power_high_dbm: float = 10.0


class VaunixLmsRfSource(RfSourceBase):
    def __init__(self, config: VaunixLmsConfig, *, library: LmsLibrary | None = None) -> None:
        self.config = config
        self._library = (
            library if library is not None else CtypesLmsLibrary(config.dll_path)
        )
        self._handle = self._library.open_device(int(config.serial))
        super().__init__(
            identity=f"vaunix-lms:{int(config.serial)}",
            frequency_low_hz=config.frequency_low_hz,
            frequency_high_hz=config.frequency_high_hz,
            power_low_dbm=config.power_low_dbm,
            power_high_dbm=config.power_high_dbm,
        )

    # ------------------------------------------------------- transport verbs
    def _write_frequency(self, value_hz: float) -> float:
        snap_to_grid(
            value_hz, FREQUENCY_UNIT_HZ, name="frequency_hz", unit="Hz"
        )
        self._library.set_frequency(
            self._handle, round(value_hz / FREQUENCY_UNIT_HZ)
        )
        return self._read_frequency()

    def _write_power(self, value_dbm: float) -> float:
        snap_to_grid(value_dbm, POWER_UNIT_DBM, name="power_dbm", unit="dBm")
        self._library.set_power(
            self._handle, round(value_dbm / POWER_UNIT_DBM)
        )
        return self._read_power()

    def _write_output(self, enabled: bool) -> bool:
        self._library.set_rf_on(self._handle, enabled)
        return self._read_output()

    def _read_frequency(self) -> float:
        return float(self._library.get_frequency(self._handle)) * FREQUENCY_UNIT_HZ

    def _read_power(self) -> float:
        return float(self._library.get_power(self._handle)) * POWER_UNIT_DBM

    def _read_output(self) -> bool:
        return bool(self._library.get_rf_on(self._handle))

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
