"""VISA and SCPI, as any instrument on this bench speaks them.

A VISA session, the resources attached to this machine, the ``*IDN?``
answer's four fields and which resource classes a probe may open are facts
about the BUS, not about one instrument.  They lived in the Rigol driver,
which is where they were first needed, so a Tektronix scope had to import
a function generator to find a scope.  A device family's driver imports
them here instead, and neither instrument knows the other exists.
"""

from __future__ import annotations

from typing import Protocol


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


#: Resource classes the probe will open.  VISA also lists ASRL serial ports,
#: and on this bench one of them is the pulse streamer's UART: opening
#: it to ask *IDN? would take the board's port from the server that owns it
#: and get nothing back, so a scan for a signal generator must never touch
#: one.  GPIB/PXI/VXI are absent for the plainer reason that nothing here has
#: ever been on one; add the prefix when something is.
PROBED_RESOURCE_PREFIXES = ("USB", "TCPIP")

#: How long one instrument may take to open and answer.  Short on purpose:
#: the probe walks every candidate in turn, and the whole family shares one
#: scan deadline, so a dead address must cost about a second, not five.
PROBE_TIMEOUT_SECONDS = 1.0

def identity_fields(identity: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(identity).split(","))



def probeable_resources(listed: object) -> tuple[str, ...]:
    """The listed resources worth opening, in the order VISA gave them."""

    return tuple(
        name
        for name in (str(item).strip() for item in listed)
        if name.upper().startswith(PROBED_RESOURCE_PREFIXES)
    )



__all__ = [
    "PROBED_RESOURCE_PREFIXES",
    "PROBE_TIMEOUT_SECONDS",
    "ScpiLink",
    "VisaResources",
    "VisaScpiLink",
    "identity_fields",
    "probeable_resources",
    "visa_resources",
]
