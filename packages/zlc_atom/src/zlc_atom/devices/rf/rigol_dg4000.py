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


class VisaResources(Protocol):
    """The whole VISA surface: what is attached, and a session on one of them."""

    def list_resources(self) -> tuple[str, ...]: ...

    def open_resource(self, resource: str, **kwargs: object) -> ScpiLink: ...


def visa_resources() -> VisaResources:
    """This machine's VISA, or why this interpreter has none.

    One entry point, because "there is no VISA here" is the same fact for
    the driver opening one named instrument and for the probe asking what is
    attached.  What it must NOT be is one sentence for every way of failing:
    this said "no VISA backend is available: install pyvisa-py" whether the
    backend was missing or PyVISA itself had never been installed, so an
    operator who had just installed both read an instruction to install what
    they had.  Which interpreter is asking is part of the answer, because
    "installed" is only ever true of one of them.
    """

    import sys

    try:
        import pyvisa
    except Exception as error:
        raise RuntimeError(
            f"PyVISA is not installed for {sys.executable}: run "
            "bin\\install_requirements.bat with THIS interpreter, or "
            f"`pip install PyVISA PyVISA-py` into it ({type(error).__name__}: {error})"
        ) from error
    try:
        return pyvisa.ResourceManager()
    except Exception as error:
        from pyvisa.highlevel import list_backends

        try:
            backends = ", ".join(list_backends()) or "none"
        except Exception:  # noqa: BLE001 - the first failure is the one to report
            backends = "unknown"
        raise RuntimeError(
            f"PyVISA {pyvisa.__version__} is installed for {sys.executable} "
            f"but no backend answered (it offers: {backends}); 'ivi' means a "
            "system NI-VISA whose visa32/visa64 DLL was not found, so install "
            "NI-VISA, or `pip install PyVISA-py` into that same interpreter "
            f"({type(error).__name__}: {error})"
        ) from error


class VisaScpiLink:
    """A pyvisa resource behind the three-verb link."""

    def __init__(self, resource: str, *, timeout_seconds: float = 5.0) -> None:
        if not isinstance(resource, str) or not resource.strip():
            raise ValueError("VISA resource name is required")
        self._resource = visa_resources().open_resource(resource.strip())
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

#: Resource classes the probe will open.  VISA also lists ASRL serial ports,
#: and on this bench one of them is the pulse streamer's 3 Mbaud UART: opening
#: it to ask *IDN? would take the board's port from the server that owns it
#: and get nothing back, so a scan for a signal generator must never touch
#: one.  GPIB/PXI/VXI are absent for the plainer reason that nothing here has
#: ever been on one; add the prefix when something is.
PROBED_RESOURCE_PREFIXES = ("USB", "TCPIP")

#: How long one instrument may take to open and answer.  Short on purpose:
#: the probe walks every candidate in turn, and the whole family shares one
#: scan deadline, so a dead address must cost about a second, not five.
PROBE_TIMEOUT_SECONDS = 1.0

#: ``*IDN?`` answers ``manufacturer,model,serial,firmware``.  The driver is
#: written for the DG4000 series -- two channels, this SCPI vocabulary -- so
#: that is what it may claim to have found.
_IDENTITY_VENDOR = "RIGOL"
_IDENTITY_MODEL_PREFIX = "DG4"


def identity_fields(identity: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(identity).split(","))


def is_dg4000(identity: str) -> bool:
    """Whether this ``*IDN?`` answer is an instrument this driver can drive."""

    fields = identity_fields(identity)
    if len(fields) < 2:
        return False
    return (
        _IDENTITY_VENDOR in fields[0].upper()
        and fields[1].upper().startswith(_IDENTITY_MODEL_PREFIX)
    )


def _identity_serial(identity: str) -> str:
    fields = identity_fields(identity)
    return fields[2] if len(fields) > 2 else ""


@dataclass(frozen=True)
class Dg4000Sighting:
    """One instrument that answered, said what it was, and was let go."""

    resource: str
    identity: str

    @property
    def serial(self) -> str:
        return _identity_serial(self.identity)

    @property
    def model(self) -> str:
        fields = identity_fields(self.identity)
        return fields[1] if len(fields) > 1 else ""


def discover_dg4000(
    resources: VisaResources | None = None,
    *,
    timeout_seconds: float = PROBE_TIMEOUT_SECONDS,
) -> tuple[Dg4000Sighting, ...]:
    """Every DG4000 attached to this machine, found by asking.

    A Lab Brick can be counted without being opened; a SCPI instrument
    cannot.  VISA lists resource NAMES -- a USB address, a socket -- and only
    ``*IDN?`` says what is on the other end, so finding one means opening a
    session, asking the one universal question, and closing it again.  That
    is the same question NI MAX asks when it populates its tree, and it is
    the reason this scan is not free: it briefly opens instruments that turn
    out to be something else.

    Everything that does not answer -- busy, held by another program, not
    SCPI at all, silent until its timeout -- is passed over.  A resource
    failing to identify itself is the ordinary case on a shared bus, not an
    error worth stopping a scan for; what IS worth stopping for is having no
    VISA at all, which ``visa_resources`` raises as an instruction.
    """

    manager = visa_resources() if resources is None else resources
    milliseconds = max(1, int(float(timeout_seconds) * 1000.0))
    found: list[Dg4000Sighting] = []
    for resource in manager.list_resources():
        name = str(resource).strip()
        if not name.upper().startswith(PROBED_RESOURCE_PREFIXES):
            continue
        try:
            session = manager.open_resource(name, open_timeout=milliseconds)
        except Exception:
            continue
        try:
            session.timeout = milliseconds
            identity = str(session.query("*IDN?")).strip()
        except Exception:
            continue
        finally:
            try:
                session.close()
            except Exception:
                pass
        if is_dg4000(identity):
            found.append(Dg4000Sighting(name, identity))
    return tuple(found)


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


__all__ = [
    "Dg4000Sighting",
    "PROBED_RESOURCE_PREFIXES",
    "PROBE_TIMEOUT_SECONDS",
    "RigolDg4000Config",
    "RigolDg4000RfSource",
    "ScpiLink",
    "VisaResources",
    "VisaScpiLink",
    "discover_dg4000",
    "is_dg4000",
    "visa_resources",
]
