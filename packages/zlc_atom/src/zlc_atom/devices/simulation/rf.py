"""The virtual RF source: the Vaunix driver over an in-memory library.

Per the bench's one virtualization rule, the fake is the LOWEST layer only:
this is the real ``VaunixLmsRfSource`` -- its unit conversions, its grid
refusals, its read-back discipline, its epochs -- driven over a library
that keeps the registers in a dict instead of a USB brick.  A detuning scan
rehearsed against this device exercises every line the real one will run.
"""

from __future__ import annotations

from zlc_atom.devices.rf.vaunix_lms import (
    VaunixLmsConfig,
    VaunixLmsRfSource,
)


class InMemoryLmsLibrary:
    """The vendor DLL's slice, holding its integers in memory."""

    def __init__(self, serials: tuple[int, ...] = (1001,)) -> None:
        self._serials = tuple(int(serial) for serial in serials)
        self._open: dict[int, dict[str, int | bool]] = {}

    def device_serials(self) -> tuple[int, ...]:
        return self._serials

    def open_device(self, serial: int) -> int:
        if int(serial) not in self._serials:
            raise LookupError(f"no Vaunix LMS with serial {serial} is attached")
        handle = int(serial)
        self._open[handle] = {"frequency": 0, "power": 0, "rf_on": False}
        return handle

    def _registers(self, handle: int) -> dict:
        registers = self._open.get(int(handle))
        if registers is None:
            raise RuntimeError("virtual LMS handle is not open")
        return registers

    def close_device(self, handle: int) -> None:
        self._open.pop(int(handle), None)

    def set_frequency(self, handle: int, frequency_units: int) -> None:
        self._registers(handle)["frequency"] = int(frequency_units)

    def get_frequency(self, handle: int) -> int:
        return int(self._registers(handle)["frequency"])

    def set_power(self, handle: int, power_units: int) -> None:
        self._registers(handle)["power"] = int(power_units)

    def get_power(self, handle: int) -> int:
        return int(self._registers(handle)["power"])

    def set_rf_on(self, handle: int, enabled: bool) -> None:
        self._registers(handle)["rf_on"] = bool(enabled)

    def get_rf_on(self, handle: int) -> bool:
        return bool(self._registers(handle)["rf_on"])


def virtual_rf_source(config: VaunixLmsConfig) -> VaunixLmsRfSource:
    """The production driver, with only its USB layer replaced."""

    return VaunixLmsRfSource(
        config, library=InMemoryLmsLibrary((int(config.serial),))
    )


__all__ = ["InMemoryLmsLibrary", "virtual_rf_source"]
