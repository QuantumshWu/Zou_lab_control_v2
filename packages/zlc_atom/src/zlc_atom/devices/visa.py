"""VISA and SCPI, as any instrument on this bench speaks them.

A VISA session, the resources attached to this machine, the ``*IDN?``
answer's four fields and which resource classes a probe may open are facts
about the BUS, not about one instrument.  They lived in the Rigol driver,
which is where they were first needed, so a Tektronix scope had to import
a function generator to find a scope.  A device family's driver imports
them here instead, and neither instrument knows the other exists.
"""

from __future__ import annotations

import threading
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

    def resource_info(self, resource: str) -> object:
        """What VISA parses ``resource`` to; its ``resource_name`` is the
        canonical spelling, which a typed address and a listed one share."""


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
        manager = visa_resources()
        held = str(manager.resource_info(resource.strip()).resource_name)
        # Held under the probe lock, opened outside it: a Scan walking the
        # bus right now finishes before the name is held, and every later
        # walk finds it held and passes it over.  The open itself must not
        # take the lock -- Init opens every device at once, and a LAN
        # instrument that is off would make the other VISA device wait out
        # its whole connect timeout before starting its own.
        with _PROBE_LOCK:
            _HELD_RESOURCES.append(held)
        try:
            self._resource = manager.open_resource(resource.strip())
        except BaseException:
            _HELD_RESOURCES.remove(held)
            raise
        self._held: str | None = held
        self._resource.timeout = int(float(timeout_seconds) * 1000.0)

    def write(self, command: str) -> None:
        self._resource.write(command)

    def query(self, command: str) -> str:
        return str(self._resource.query(command))

    def close(self) -> None:
        try:
            self._resource.close()
        finally:
            # Not under the probe lock: a close must not wait out a Scan's
            # walk, and a name let go mid-walk is at worst passed over once.
            if self._held is not None:
                _HELD_RESOURCES.remove(self._held)
                self._held = None


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

#: One walk of the bus at a time.  A Scan asks every family at once, each
#: from its own thread, and the DG4000 and the Tek scope both find theirs by
#: walking this same list: two sessions asking one instrument ``*IDN?`` at
#: the same moment can interrupt each other's query or fail a USB claim, and
#: the walk that lost passes the instrument over as silent.  A device's own
#: session marks its instrument held under it before opening, so no walk is
#: ever mid-way through asking the instrument a device is opening; the open
#: itself runs outside it, so two devices never take turns opening.
_PROBE_LOCK = threading.Lock()

#: The canonical names of the resources this process holds open as devices,
#: once per open session.  A walk passes them over: a device's session may be
#: mid-conversation with that instrument -- a capture polling, a tune waiting
#: for its answer -- and a second session asking ``*IDN?`` under it is the
#: same interrupted query the lock prevents between two walks.  A held
#: instrument is already configured, so the Scan loses nothing by not asking.
_HELD_RESOURCES: list[str] = []


def identity_fields(identity: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(identity).split(","))


def probeable_resources(listed: object) -> tuple[str, ...]:
    """The listed resources worth opening, in the order VISA gave them."""

    return tuple(
        name
        for name in (str(item).strip() for item in listed)
        if name.upper().startswith(PROBED_RESOURCE_PREFIXES)
    )


def identify_resources(
    resources: VisaResources | None = None,
    *,
    timeout_seconds: float = PROBE_TIMEOUT_SECONDS,
) -> tuple[tuple[str, str], ...]:
    """Every probeable resource that answered ``*IDN?``, with its answer.

    A SCPI instrument cannot be counted without being opened.  VISA lists
    resource NAMES -- a USB address, a socket -- and only ``*IDN?`` says
    what is on the other end, so finding one means opening a session,
    asking the one universal question, and closing it again: what NI MAX
    does to populate its tree, and why a scan briefly opens instruments
    that turn out to be something else.

    Everything that does not answer -- busy, held by another program, not
    SCPI at all, silent until its timeout -- is passed over.  A resource
    failing to identify itself is the ordinary case on a shared bus, not an
    error worth stopping a scan for.  What IS worth stopping for is having no
    VISA at all, which ``visa_resources`` raises as an instruction, and a
    VISA that lists nothing to ask, raised here: "found nothing" is only an
    answer if something was asked.

    Every family that looks for its instrument this way takes its turn under
    ``_PROBE_LOCK``, and an instrument this process holds open as a device is
    not asked at all, so no instrument is ever asked twice at once.
    """

    with _PROBE_LOCK:
        manager = visa_resources() if resources is None else resources
        listed = tuple(str(name) for name in manager.list_resources())
        probeable = probeable_resources(listed)
        if not probeable:
            # VISA's own list is far blinder than an operator expects: a LAN
            # instrument appears only once it has been added in NI MAX, and a
            # USB one only once its USB-TMC driver is bound -- so an
            # instrument sitting there, plugged in and working, can simply
            # not be in the list.  Saying nothing then reports "none here"
            # about a bench that has one.
            raise RuntimeError(
                "VISA lists nothing to ask: no "
                f"{' or '.join(PROBED_RESOURCE_PREFIXES)} resource is registered "
                f"on this machine (it lists: {', '.join(listed) or 'nothing'}). "
                "A LAN instrument has to be added in NI MAX -- or skip that and "
                "type its TCPIP0::<address>::INSTR in by hand, which needs no "
                "install; a USB one is invisible to VISA until a USB-TMC driver "
                "is bound to it, which is what installing NI-VISA (or the "
                "instrument maker's own VISA package, such as Rigol UltraSigma) "
                "does."
            )
        milliseconds = max(1, int(float(timeout_seconds) * 1000.0))
        found: list[tuple[str, str]] = []
        for name in probeable:
            try:
                if (
                    _HELD_RESOURCES
                    and str(manager.resource_info(name).resource_name)
                    in _HELD_RESOURCES
                ):
                    continue
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
            found.append((name, identity))
        return tuple(found)


__all__ = [
    "PROBED_RESOURCE_PREFIXES",
    "PROBE_TIMEOUT_SECONDS",
    "ScpiLink",
    "VisaResources",
    "VisaScpiLink",
    "identify_resources",
    "identity_fields",
    "probeable_resources",
    "visa_resources",
]
