"""Rigol DG4000-series function generator as an RF source, over SCPI.

The instrument speaks SCPI text over VISA (USB-TMC or LAN), and the driver
is written against a three-verb link so the transport is the ONLY thing a
test or a virtual bench has to stand in for -- the SCPI vocabulary, the
read-back discipline and the bound checks are all exercised as shipped.

Frequency is written and read in hertz; power in dBm (the channel's
amplitude unit is pinned to DBM once at open, so a front-panel change
cannot silently re-interpret every later write).  ``tune`` returns what the
instrument reports back, never what was asked -- the scan executor refuses
any difference, which is how a mistyped bound or a loading-dependent
amplitude shows up as a named error instead of a wrong dataset column.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from zlc_atom.devices.rf.contract import RfSourceBase


class ScpiLink(Protocol):
    """The whole transport surface a SCPI instrument needs."""

    def write(self, command: str) -> None: ...

    def query(self, command: str) -> str: ...

    def close(self) -> None: ...


class VisaScpiLink:
    """A pyvisa resource behind the three-verb link."""

    def __init__(self, resource: str, *, timeout_seconds: float = 5.0) -> None:
        if not isinstance(resource, str) or not resource.strip():
            raise ValueError("VISA resource name is required")
        try:
            import pyvisa

            manager = pyvisa.ResourceManager()
        except Exception as error:
            raise RuntimeError(
                "no VISA backend is available: install NI-VISA system-wide "
                "(or `pip install pyvisa-py`), then restart the bench "
                f"({type(error).__name__}: {error})"
            ) from error
        self._resource = manager.open_resource(resource.strip())
        self._resource.timeout = int(float(timeout_seconds) * 1000.0)

    def write(self, command: str) -> None:
        self._resource.write(command)

    def query(self, command: str) -> str:
        return str(self._resource.query(command))

    def close(self) -> None:
        self._resource.close()


@dataclass(frozen=True)
class RigolDg4000Config:
    """Where the instrument is and any policy window this bench imposes.

    Every edge is optional.  ``None`` delegates only to the instrument's own
    physical limits; setting an edge adds a bench policy limit and also gives
    the scan authoring surface that side of its finite range.  The channel
    count is a fact about the SERIES -- every DG4000 has two -- so it is not
    authored at all: one instrument is one installed instance, its channels
    are its own knobs.
    """

    resource: str
    frequency_low_hz: float | None = None
    frequency_high_hz: float | None = None
    power_low_dbm: float | None = None
    power_high_dbm: float | None = None
    timeout_seconds: float = 5.0


#: ch1 -> :SOURce1/:OUTPut1.  The channel NAMES are field-name prefixes
#: (ch1_frequency_hz), the numbers are SCPI's.
_CHANNELS = ("ch1", "ch2")


class RigolDg4000RfSource(RfSourceBase):
    def __init__(self, config: RigolDg4000Config, *, link: ScpiLink | None = None) -> None:
        self.config = config
        self._link = link if link is not None else VisaScpiLink(
            config.resource, timeout_seconds=config.timeout_seconds
        )
        identity = self._link.query("*IDN?").strip()
        if not identity:
            raise RuntimeError("the instrument answered *IDN? with nothing")
        # Pin the amplitude unit once, per channel: every later write and
        # read of power means dBm, whatever the front panel was showing.
        for channel in _CHANNELS:
            self._link.write(f"{self._source(channel)}:VOLTage:UNIT DBM")
        super().__init__(
            identity=identity,
            channels=_CHANNELS,
            frequency_low_hz=config.frequency_low_hz,
            frequency_high_hz=config.frequency_high_hz,
            power_low_dbm=config.power_low_dbm,
            power_high_dbm=config.power_high_dbm,
        )

    @staticmethod
    def _source(channel: str) -> str:
        return f":SOURce{channel[2:]}"

    @staticmethod
    def _output(channel: str) -> str:
        return f":OUTPut{channel[2:]}"

    # ------------------------------------------------------- transport verbs
    def _write_frequency(self, channel: str, value_hz: float) -> float:
        self._link.write(f"{self._source(channel)}:FREQuency {value_hz:.6f}")
        return self._read_frequency(channel)

    def _write_power(self, channel: str, value_dbm: float) -> float:
        self._link.write(f"{self._source(channel)}:VOLTage {value_dbm:.4f}")
        return self._read_power(channel)

    def _write_output(self, channel: str, enabled: bool) -> bool:
        self._link.write(
            f"{self._output(channel)} {'ON' if enabled else 'OFF'}"
        )
        return self._read_output(channel)

    def _read_frequency(self, channel: str) -> float:
        return float(self._link.query(f"{self._source(channel)}:FREQuency?"))

    def _read_power(self, channel: str) -> float:
        return float(self._link.query(f"{self._source(channel)}:VOLTage?"))

    def _read_output(self, channel: str) -> bool:
        answer = self._link.query(f"{self._output(channel)}?").strip().upper()
        return answer in ("ON", "1")

    def close(self) -> None:
        self._link.close()


__all__ = ["RigolDg4000Config", "RigolDg4000RfSource", "ScpiLink", "VisaScpiLink"]
