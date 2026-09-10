"""Rigol DG4000-series function generator as an RF source, over SCPI.

The instrument speaks SCPI text over VISA (USB-TMC or LAN), and the driver
is written against a three-verb link so the transport is the ONLY thing a
test or a virtual bench has to stand in for -- the SCPI vocabulary, the
read-back discipline and the bound checks are all exercised as shipped.

Frequency is written and read in hertz; power's canonical unit is dBm.
Connecting and explicit Refresh read the channel's unit, load, waveform,
limits and values. Apply uses those bounded session facts, writes the
selected quantity, and returns the instrument's actual readback. External
front-panel changes are adopted by Refresh, not by probing the whole
instrument before every scan point.

Explicit selected-unit Apply uses ``tune_in_unit``: it selects the channel's
native amplitude unit only when needed and writes volts directly. A raw
``read_tunable_in_unit`` snapshot lets Scan restore both number and unit.

Frequency-dependent amplitude limits and current become unknown after a
frequency write. They are read again when amplitude is next needed; the
driver does not undo a requested frequency to counter the instrument's own
amplitude limiting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Protocol

from zlc_atom.devices.rf.contract import POWER_FIELD, WINDOW_FIELDS, RfSourceBase, channel_field
from zlc_atom.authoring import AuthoringField, TunableField
from zlc_data.units import DEFAULT_UNITS


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
#: (ch1_frequency), the numbers are SCPI's.
_CHANNELS = ("ch1", "ch2")
#: The three amplitude units a DG4000 channel can be displaying.
_DBM = "DBM"
_VRMS = "VRMS"
_VPP = "VPP"
#: One milliwatt, the reference the "dB" in dBm is measured from.
_MILLIWATT = 1e-3
#: The one waveform whose peak-to-peak/RMS ratio (2*sqrt(2)) this driver
#: knows; ``:FUNCtion?`` answers it as SIN (or its long form).
_SINE = "SIN"
_SINE_VPP_PER_VRMS = 2.0 * math.sqrt(2.0)
#: The instrument spells high-Z as this out-of-range ohm count.
_HIGH_Z_OHMS = 1e6

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


def probeable_resources(listed: object) -> tuple[str, ...]:
    """The listed resources worth opening, in the order VISA gave them."""

    return tuple(
        name
        for name in (str(item).strip() for item in listed)
        if name.upper().startswith(PROBED_RESOURCE_PREFIXES)
    )


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
    for name in probeable_resources(manager.list_resources()):
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
        # The authored half first, with nothing open: a window that cannot
        # be honoured is refused before a VISA session exists to leak.
        super().__init__(
            channels=_CHANNELS,
            frequency_low_hz=config.frequency_low_hz,
            frequency_high_hz=config.frequency_high_hz,
            power_low_dbm=config.power_low_dbm,
            power_high_dbm=config.power_high_dbm,
        )
        self._link = link if link is not None else VisaScpiLink(
            config.resource, timeout_seconds=config.timeout_seconds
        )
        self._amplitudes: dict[str, dict[str, object]] = {}
        # From here on the session is this object's to close: a failure
        # before the constructor returns has no other owner to hand it to.
        try:
            identity = self._link.query("*IDN?").strip()
            if not identity:
                raise RuntimeError("the instrument answered *IDN? with nothing")
            self._attach(identity)
        except BaseException as error:
            try:
                self._link.close()
            except BaseException as close_error:
                error.add_note(
                    "closing the VISA session also reported: "
                    f"{type(close_error).__name__}: {close_error}"
                )
            raise

    @staticmethod
    def _source(channel: str) -> str:
        return f":SOURce{channel[2:]}"

    @staticmethod
    def _output(channel: str) -> str:
        return f":OUTPut{channel[2:]}"

    # ------------------------------------------------------- transport verbs
    def _write_frequency(self, channel: str, value_hz: float) -> float:
        """Set and read the requested field, without unrelated queries."""
        source = self._source(channel)
        self._current_values.pop(channel_field(channel, POWER_FIELD), None)
        if channel in self._amplitudes:
            self._amplitudes[channel].update(current=None, minimum=None, maximum=None)
        self._device_limits[channel] = (self._device_limits[channel][0], None)
        self._link.write(f"{source}:FREQuency {value_hz:.17g}")
        return self._read_frequency(channel)

    def _write_power(self, channel: str, value_dbm: float) -> float:
        held = self._amplitude_state(channel)
        unit = held["unit"]
        if unit is None:
            unit = held["unit"] = self._amplitude_unit(channel)
        target = {_DBM: "dBm", _VPP: "Vpp", _VRMS: "Vrms"}[unit]
        amplitude = self._convert_amplitude(channel, value_dbm, "dBm", target)
        actual = self._write_power_in_unit(channel, amplitude, target)
        return self._convert_amplitude(channel, actual, target, "dBm")

    def _amplitude_state(self, channel: str) -> dict[str, object]:
        """The one bounded set of channel facts, refreshed only when needed."""

        held = self._amplitudes.get(channel)
        if held is None:
            source = self._source(channel)
            held = {
                "unit": self._amplitude_unit(channel),
                "load": self._load_ohms(channel),
                "waveform": self._link.query(f"{source}:FUNCtion?").strip().upper(),
                "current": float(self._link.query(f"{source}:VOLTage?")),
                "minimum": float(self._link.query(f"{source}:VOLTage? MINimum")),
                "maximum": float(self._link.query(f"{source}:VOLTage? MAXimum")),
            }
            self._amplitudes[channel] = held
        return held

    def tunable_values(self) -> dict[str, object]:
        with self._condition:
            self._amplitudes.clear()
            values = super().tunable_values()
            for channel in self._channels:
                frequency_limits, _power_limits = self._device_limits[channel]
                self._device_limits[channel] = (frequency_limits, self._read_power_limits(channel))
            return values

    @staticmethod
    def _amplitude_unit_parts(unit: str) -> tuple[str | None, float]:
        family, prefix = DEFAULT_UNITS.family_and_prefix(unit)
        native = {"dBm": _DBM, "Vpp": _VPP, "Vrms": _VRMS}.get(family.symbol)
        return native, 10.0 ** prefix.exponent

    def _check_policy_units(self, name: str, *units: str) -> None:
        policy_unit = next((unit for field, _label, unit, _config_name in WINDOW_FIELDS if field == name), None)
        if policy_unit is None or DEFAULT_UNITS.resolve(policy_unit).dimension != "power":
            return
        if any(unit and self._amplitude_unit_parts(unit)[0] in (_VPP, _VRMS) for unit in units):
            raise ValueError(
                f"{name}: shared power policy has no single channel load; use dBm or W/mW, not voltage units"
            )

    def _convert_amplitude(self, channel: str, value: float, source: str, target: str) -> float:
        """Convert with the channel's load/waveform; prefixes never visit dBm."""
        source_native, source_scale = self._amplitude_unit_parts(source)
        target_native, target_scale = self._amplitude_unit_parts(target)
        if source == target:
            return float(value)
        if source_native is None:
            dbm = float(DEFAULT_UNITS.convert(value, source, "dBm"))
            return self._convert_amplitude(channel, dbm, "dBm", target)
        if target_native is None:
            dbm = self._convert_amplitude(channel, value, source, "dBm")
            return float(DEFAULT_UNITS.convert(dbm, "dBm", target))
        raw = float(value) * source_scale
        if source_native == target_native:
            return raw / target_scale
        if source_native == _DBM:
            raw = self._volts_from_dbm(channel, raw, target_native)
        elif target_native == _DBM:
            raw = self._dbm_from_volts(channel, raw, source_native)
        elif source_native == _VPP:
            raw /= self._vpp_per_vrms(channel)
        else:
            raw *= self._vpp_per_vrms(channel)
        return raw / target_scale

    def convert_tunable_value(
        self, name: str, value: float | tuple | list, source_unit: str, target_unit: str
    ) -> float | tuple[float, ...]:
        """Read-only conversion through the same channel facts used by Apply."""
        routed = self._routing.get(str(name))
        if routed is None and str(name) not in {field[0] for field in WINDOW_FIELDS}:
            raise ValueError(f"device has no tunable field {name!r}")
        self._check_policy_units(str(name), source_unit, target_unit)
        with self._condition:
            def converted(item):
                if routed is not None and routed[1] == POWER_FIELD:
                    return self._convert_amplitude(routed[0], float(item), source_unit, target_unit)
                return float(DEFAULT_UNITS.convert(item, source_unit, target_unit))
            return tuple(converted(item) for item in value) if isinstance(value, (tuple, list)) else converted(value)

    def read_tunable_in_unit(self, name: str, unit: str = "") -> TunableField:
        """Read a selected-unit view, retaining raw standing-unit readback."""
        routed = self._routing.get(str(name))
        self._check_policy_units(str(name), unit)
        with self._condition:
            if routed is None or routed[1] != POWER_FIELD:
                field = next((item for item in self.tunable_fields() if item.metadata.name == name), None)
                if field is None:
                    raise ValueError(f"device has no tunable field {name!r}")
                source = field.metadata.unit or "1"
                target = unit or source
                if source == target:
                    return field
                def converted(value):
                    return None if value is None else float(DEFAULT_UNITS.convert(value, source, target))
                return replace(field, metadata=replace(field.metadata, unit=target,
                    default=converted(field.metadata.default), minimum=converted(field.metadata.minimum),
                    maximum=converted(field.metadata.maximum)), current=converted(field.current),
                    device_limits=None if field.device_limits is None else tuple(converted(v) for v in field.device_limits))
            channel = routed[0]
            held = self._amplitude_state(channel)
            native = held["unit"]
            standing_unit = {_DBM: "dBm", _VPP: "Vpp", _VRMS: "Vrms"}.get(native, "dBm")
            target = unit or standing_unit
            raw = held["current"]
            current = (raw if raw is None or target == standing_unit
                       else self._convert_amplitude(channel, raw, standing_unit, target))
            limits = None if held["minimum"] is None else self._instrument_limits(tuple(
                self._convert_amplitude(channel, float(held[edge]), standing_unit, target)
                for edge in ("minimum", "maximum")
            ), name=str(name), unit=target)
            policy = tuple(None if edge is None else self._convert_amplitude(channel, edge, "dBm", target)
                           for edge in self._power_bounds)
            low, high = (policy if limits is None else
                         self._effective_range(policy, limits, name=str(name), unit=target))
            if raw is not None:
                self._current_values[str(name)] = self._convert_amplitude(channel, raw, standing_unit, "dBm")
            return TunableField(
                AuthoringField(str(name), "float", f"{self._channel_label(channel)}Power", None,
                               minimum=low, maximum=high, unit=target),
                current, True, (str(name),), device_limits=limits,
            )

    def tune_in_unit(self, name: str, value: float, unit: str) -> float:
        routed = self._routing.get(str(name))
        if not unit or value is None:
            return self.tune(name, value)
        self._check_policy_units(str(name), unit)
        if routed is not None and routed[1] == POWER_FIELD:
            return self._tune_logged(name, value, unit=unit)
        field = self.read_tunable_in_unit(name)
        native = field.metadata.unit or "1"
        if unit == native:
            return self.tune(name, value)
        requested = float(DEFAULT_UNITS.convert(value, unit, native))
        actual = self.tune(name, requested)
        return float(DEFAULT_UNITS.convert(actual, native, unit))

    def _write_power_in_unit(self, channel: str, value: float, unit: str) -> float:
        native, scale = self._amplitude_unit_parts(unit)
        if native is None:
            # W/mW have no DG4000 display mode. Keep its mode and use the
            # existing canonical boundary, with this driver's load conversion.
            actual = self._write_power(channel, float(DEFAULT_UNITS.convert(value, unit, "dBm")))
            return float(DEFAULT_UNITS.convert(actual, "dBm", unit))
        source = self._source(channel)
        held = self._amplitude_state(channel)
        old_unit = held["unit"]
        old_value = held["current"]
        try:
            if native != old_unit:
                self._link.write(f"{source}:VOLTage:UNIT {native}")
            self._link.write(f"{source}:VOLTage {float(value) * scale:.17g}{native}")
            actual = float(self._link.query(f"{source}:VOLTage?"))
            old_symbol = {_DBM: "dBm", _VPP: "Vpp", _VRMS: "Vrms"}.get(old_unit, "dBm")
            new_symbol = {_DBM: "dBm", _VPP: "Vpp", _VRMS: "Vrms"}[native]
            for edge in ("minimum", "maximum"):
                if held[edge] is not None:
                    held[edge] = self._convert_amplitude(channel, float(held[edge]), old_symbol, new_symbol)
            held.update(unit=native, current=actual)
            return actual / scale
        except BaseException as error:
            # Control Apply also owns a complete command: a failed write
            # must not leave its temporary unit behind. Restore raw numbers,
            # never a dBm round trip, and preserve both failures if it cannot.
            commands = ([f"{source}:VOLTage:UNIT {old_unit}"]
                        if old_unit is not None and native != old_unit else [])
            if old_value is not None:
                commands.append(f"{source}:VOLTage {old_value:.17g}{old_unit}")
            for command in commands:
                try:
                    self._link.write(command)
                except BaseException as restore_error:
                    error.add_note(f"restoring channel {channel} also failed: {restore_error}")
            held.update(unit=None, current=None, minimum=None, maximum=None)
            self._device_limits[channel] = (self._device_limits[channel][0], None)
            self._current_values.pop(channel_field(channel, POWER_FIELD), None)
            raise

    def _write_output(self, channel: str, enabled: bool) -> bool:
        self._link.write(
            f"{self._output(channel)} {'ON' if enabled else 'OFF'}"
        )
        return self._read_output(channel)

    def _read_frequency(self, channel: str) -> float:
        return float(self._link.query(f"{self._source(channel)}:FREQuency?"))

    def _read_frequency_limits(self, channel: str) -> tuple[float, float]:
        """The instrument's own range, asked of it: a DG4062 stops at 60 MHz
        and a DG4162 at 160 MHz, and the model knows which it is."""

        source = self._source(channel)
        return (
            float(self._link.query(f"{source}:FREQuency? MINimum")),
            float(self._link.query(f"{source}:FREQuency? MAXimum")),
        )

    def _read_power_limits(self, channel: str) -> tuple[float, float]:
        """The instrument's amplitude range, in dBm through the channel's own
        unit and load -- the same arithmetic every power read goes through."""

        held = self._amplitude_state(channel)
        unit = held["unit"]
        edges = []
        for extreme in ("minimum", "maximum"):
            amplitude = float(held[extreme])
            edges.append(
                amplitude if unit == _DBM else self._dbm_from_volts(channel, amplitude, unit)
            )
        return min(edges), max(edges)

    def _read_power(self, channel: str) -> float:
        held = self._amplitude_state(channel)
        if held["current"] is None:
            self._amplitudes.pop(channel)
            held = self._amplitude_state(channel)
            self._device_limits[channel] = (self._device_limits[channel][0], self._read_power_limits(channel))
        unit = held["unit"]
        amplitude = float(held["current"])
        if unit == _DBM:
            return amplitude
        return self._dbm_from_volts(channel, amplitude, unit)

    # ---------------------------------------------- the channel's own units
    def _amplitude_unit(self, channel: str) -> str:
        """Read the unit while refreshing this channel's session facts."""

        answer = self._link.query(
            f"{self._source(channel)}:VOLTage:UNIT?"
        ).strip().upper()
        for unit in (_DBM, _VRMS, _VPP):
            if answer.startswith(unit):
                return unit
        raise RuntimeError(
            f"channel {channel} answered its amplitude unit as {answer!r}, "
            f"which is none of {_DBM}, {_VRMS} or {_VPP}"
        )

    def _load_ohms(self, channel: str) -> float:
        """The load this channel is set to drive, in ohms (inf for high-Z)."""

        answer = self._link.query(
            f"{self._output(channel)}:IMPedance?"
        ).strip().upper()
        if answer.startswith("INF"):
            return math.inf
        ohms = float(answer)
        # The instrument spells high-Z as its own out-of-range number.
        if not math.isfinite(ohms) or ohms >= _HIGH_Z_OHMS:
            return math.inf
        if ohms <= 0.0:
            raise RuntimeError(
                f"channel {channel} reports a load of {ohms!r} ohms"
            )
        return ohms

    def _delivering_load(self, channel: str, unit: str) -> float:
        load = float(self._amplitude_state(channel)["load"])
        if math.isinf(load):
            raise RuntimeError(
                f"channel {channel} states its amplitude in {unit} into a "
                "high-Z load, where delivered power is not defined; set the "
                "channel's output load, or its amplitude unit to dBm"
            )
        return load

    def _vpp_per_vrms(self, channel: str) -> float:
        """The channel's peak-to-peak/RMS ratio, which is its waveform's.

        Refreshed with the unit/load: only a sine's ratio is known here; a square wave's
        depends on its duty cycle, a ramp's on its symmetry, and reading
        either through the sine ratio would state a power the channel is
        not delivering (3 dB off for a symmetric square wave).
        """

        answer = str(self._amplitude_state(channel)["waveform"])
        if answer.startswith(_SINE):
            return _SINE_VPP_PER_VRMS
        raise RuntimeError(
            f"channel {channel} states its amplitude in {_VPP} while playing "
            f"a {answer or 'unknown'} waveform, whose peak-to-peak/RMS ratio "
            "this driver does not know; set the channel's amplitude unit to "
            f"{_VRMS} or {_DBM}, or its waveform to sine"
        )

    def _rms_from_volts(self, channel: str, amplitude: float, unit: str) -> float:
        return amplitude if unit == _VRMS else amplitude / self._vpp_per_vrms(channel)

    def _dbm_from_volts(self, channel: str, amplitude: float, unit: str) -> float:
        load = self._delivering_load(channel, unit)
        rms = self._rms_from_volts(channel, amplitude, unit)
        watts = rms * rms / load
        if watts <= 0.0:
            raise RuntimeError(
                f"channel {channel} reports an amplitude of {amplitude!r} "
                f"{unit}, which delivers no power to state in dBm"
            )
        return 10.0 * math.log10(watts / _MILLIWATT)

    def _volts_from_dbm(self, channel: str, value_dbm: float, unit: str) -> float:
        load = self._delivering_load(channel, unit)
        watts = _MILLIWATT * 10.0 ** (value_dbm / 10.0)
        rms = math.sqrt(watts * load)
        return rms if unit == _VRMS else rms * self._vpp_per_vrms(channel)

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
    "probeable_resources",
    "visa_resources",
]
