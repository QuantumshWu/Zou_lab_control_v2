"""The device-independent RF source surface every implementation answers.

An RF source is, to this system, its TUNABLE SURFACE: per output channel a
frequency, a power and an output switch, each read and written through the
same duck-typed quartet every tunable device speaks (``tunable_fields`` /
``tune`` / ``tunable_values`` / ``settings_provenance``).  A scan axis, the
generic control panel and the device-axis executor all consume exactly that
quartet, so the capability Protocol IS the quartet -- there is no second,
RF-only vocabulary for a consumer to learn.

CHANNELS ARE THE DEVICE'S OWN STRUCTURE.  One instrument is one installed
instance whatever its channel count -- a two-channel generator is not two
devices to manage, it is one device with six knobs.  A single-channel
source keeps the bare field names (``frequency_hz``); a multi-channel one
prefixes them with its own channel names (``ch1_frequency_hz``), so the
add-axis combo and the control panel show every knob of the one instrument
under its one card.

What varies between instruments is the transport underneath (SCPI text over
VISA for a bench generator, a vendor DLL for a Lab Brick) and the value grid
the hardware quantizes to.  Both live in the concrete drivers; the shared
plumbing here owns the rules that must not fork per driver:

* a write is SET-THEN-READ-BACK, and ``tune`` returns what the instrument
  itself reports, never what was asked;
* a value the instrument would silently round is REFUSED before it is
  written, naming the grid -- a scan coordinate must mean exactly what its
  dataset column says (the same law the pulse DAC axes obey);
* every accepted change advances ``settings_epoch``, so a control panel and
  a running scan can see each other's writes.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any, Mapping, Protocol, runtime_checkable

from zlc_atom.authoring import AuthoringField, TunableField

FREQUENCY_FIELD = "frequency_hz"
POWER_FIELD = "power_dbm"
OUTPUT_FIELD = "output_enabled"

#: The bench's safety window is a CONTROL knob, not an apparatus fact: the
#: operator adjusts it on the control panel (plain Apply, never live) or
#: through the same ``tune`` API, and nothing else may scan it.  It is
#: deliberately unbounded and non-live so scan_ports_for_devices never
#: offers it as an axis.  (name, label, unit) per policy field.
WINDOW_FIELDS = (
    ("frequency_low_hz", "Frequency low (Hz)", "Hz"),
    ("frequency_high_hz", "Frequency high (Hz)", "Hz"),
    ("power_low_dbm", "Power low (dBm)", "dBm"),
    ("power_high_dbm", "Power high (dBm)", "dBm"),
)

#: Interactions narrate HERE, in the contract, so every path that moves a
#: knob -- a scan's device axis, the control panel, a notebook, a remote
#: client -- leaves the same trace.  Each line ends with ``device=<identity>``
#: so a bench window can show one instrument's story and nobody else's.
_LOG = logging.getLogger(__name__)


@runtime_checkable
class RfSource(Protocol):
    """What an installed ``rf.source`` capability answers."""

    def tunable_fields(self) -> tuple[TunableField, ...]: ...

    def tune(self, name: str, value: Any) -> Any: ...

    def tunable_values(self) -> Mapping[str, Any]: ...

    def settings_provenance(self) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


def snap_to_grid(value: float, step: float, *, name: str, unit: str) -> float:
    """The value itself, or a refusal naming the instrument's grid.

    The instrument quantizes -- a Lab Brick holds frequency in 10 Hz units
    and power in quarter-dB units -- and writing a value off that grid would
    silently move what the scan believes it measured.  Refusing here, before
    the write, keeps the dataset column exactly truthful and tells the
    operator the step to author their scan on.
    """

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    quantized = round(value / step) * step
    # Compare in grid units, where representable values are exact integers;
    # comparing the floats re-manufactures the rounding noise being judged.
    if abs(value / step - round(value / step)) > 1e-9:
        raise ValueError(
            f"{name}={value!r} {unit} is not on this instrument's "
            f"{step!r} {unit} grid; nearest is {quantized!r}"
        )
    return quantized


def channel_field(channel: str, field: str) -> str:
    """The one spelling of a channel's field name, shared with consumers."""

    return f"{channel}_{field}" if channel else field


class RfSourceBase:
    """The shared half of every RF driver: channels, bounds, epochs, locking.

    A concrete driver names its channels ("" for the single-channel case,
    ``("ch1", "ch2")`` for a two-channel generator) and supplies the
    transport verbs, each taking the channel and returning the instrument's
    own read-back.  Everything the consumers see -- the tunable quartet --
    lives here once.
    """

    def __init__(
        self,
        *,
        identity: str,
        channels: tuple[str, ...] = ("",),
        frequency_low_hz: float,
        frequency_high_hz: float,
        power_low_dbm: float,
        power_high_dbm: float,
    ) -> None:
        if not (
            math.isfinite(frequency_low_hz)
            and math.isfinite(frequency_high_hz)
            and frequency_low_hz < frequency_high_hz
        ):
            raise ValueError("frequency bounds must be finite and ordered")
        if not (
            math.isfinite(power_low_dbm)
            and math.isfinite(power_high_dbm)
            and power_low_dbm < power_high_dbm
        ):
            raise ValueError("power bounds must be finite and ordered")
        names = tuple(str(channel) for channel in channels)
        if not names or len(set(names)) != len(names):
            raise ValueError("rf channels must be non-empty and unique")
        if len(names) > 1 and any(not name for name in names):
            raise ValueError("a multi-channel source names every channel")
        self._identity = str(identity)
        self._channels = names
        self._frequency_bounds = (float(frequency_low_hz), float(frequency_high_hz))
        self._power_bounds = (float(power_low_dbm), float(power_high_dbm))
        self._condition = threading.Condition()
        self._settings_epoch = 0
        #: field name -> (channel, kind); the ONE table tune() resolves by,
        #: so a field's spelling cannot drift from its routing.
        self._routing: dict[str, tuple[str, str]] = {}
        for channel in names:
            for kind in (FREQUENCY_FIELD, POWER_FIELD, OUTPUT_FIELD):
                self._routing[channel_field(channel, kind)] = (channel, kind)
        # OPENING BRINGS THE KNOBS INTO THE BENCH'S WINDOW.  The authored
        # bounds are bench policy for what a scan may command, and
        # TunableField refuses to describe a current value that stands
        # outside them -- rightly, a form cannot offer a range the truth is
        # not in.  A fresh instrument idles wherever it likes (a Lab Brick
        # register at 0 Hz, a bench generator at its power-on default), so
        # the open drives any out-of-window knob to the nearest bound, per
        # channel.  The OUTPUT switches are never touched: policy may move
        # a silent knob, never un-silence one.
        for channel in names:
            frequency = float(self._read_frequency(channel))
            if not (
                self._frequency_bounds[0]
                <= frequency
                <= self._frequency_bounds[1]
            ):
                bounded = min(
                    max(frequency, self._frequency_bounds[0]),
                    self._frequency_bounds[1],
                )
                self._write_frequency(channel, bounded)
                _LOG.info(
                    "OPEN NORMALIZED field=%s from=%r to=%r device=%s",
                    channel_field(channel, FREQUENCY_FIELD),
                    frequency,
                    bounded,
                    self._identity,
                )
            power = float(self._read_power(channel))
            if not (self._power_bounds[0] <= power <= self._power_bounds[1]):
                bounded = min(
                    max(power, self._power_bounds[0]), self._power_bounds[1]
                )
                self._write_power(channel, bounded)
                _LOG.info(
                    "OPEN NORMALIZED field=%s from=%r to=%r device=%s",
                    channel_field(channel, POWER_FIELD),
                    power,
                    bounded,
                    self._identity,
                )

    # ------------------------------------------------------- transport verbs
    def _write_frequency(self, channel: str, value_hz: float) -> float:
        raise NotImplementedError

    def _write_power(self, channel: str, value_dbm: float) -> float:
        raise NotImplementedError

    def _write_output(self, channel: str, enabled: bool) -> bool:
        raise NotImplementedError

    def _read_frequency(self, channel: str) -> float:
        raise NotImplementedError

    def _read_power(self, channel: str) -> float:
        raise NotImplementedError

    def _read_output(self, channel: str) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    # -------------------------------------------------------------- contract
    def _channel_label(self, channel: str) -> str:
        return f"{channel.upper()} · " if channel else ""

    def tunable_fields(self) -> tuple[TunableField, ...]:
        frequency_low, frequency_high = self._frequency_bounds
        power_low, power_high = self._power_bounds
        fields: list[TunableField] = []
        with self._condition:
            for channel in self._channels:
                label = self._channel_label(channel)
                fields.append(
                    TunableField(
                        metadata=AuthoringField(
                            channel_field(channel, FREQUENCY_FIELD),
                            "float",
                            f"{label}Frequency (Hz)",
                            frequency_low,
                            minimum=frequency_low,
                            maximum=frequency_high,
                            unit="Hz",
                        ),
                        current=float(self._read_frequency(channel)),
                        live_write=True,
                        dependency_group=(
                            channel_field(channel, FREQUENCY_FIELD),
                        ),
                    )
                )
                fields.append(
                    TunableField(
                        metadata=AuthoringField(
                            channel_field(channel, POWER_FIELD),
                            "float",
                            f"{label}Power (dBm)",
                            power_low,
                            minimum=power_low,
                            maximum=power_high,
                            unit="dBm",
                        ),
                        current=float(self._read_power(channel)),
                        live_write=True,
                        dependency_group=(channel_field(channel, POWER_FIELD),),
                    )
                )
                # No bounds on purpose: a bool is a switch, not a scan axis,
                # and scan_ports_for_devices admits only bounded fields -- so
                # the output toggles appear on the control panel and never in
                # the add-axis combo.
                fields.append(
                    TunableField(
                        metadata=AuthoringField(
                            channel_field(channel, OUTPUT_FIELD),
                            "bool",
                            f"{label}Output enabled",
                            False,
                        ),
                        current=bool(self._read_output(channel)),
                        live_write=True,
                        dependency_group=(channel_field(channel, OUTPUT_FIELD),),
                    )
                )
            for name, label, unit in WINDOW_FIELDS:
                current = float(self._window_value(name))
                fields.append(
                    TunableField(
                        metadata=AuthoringField(
                            name, "float", label, current, unit=unit
                        ),
                        current=current,
                        live_write=False,
                        dependency_group=(name,),
                    )
                )
        return tuple(fields)

    def tunable_values(self) -> dict[str, Any]:
        with self._condition:
            values: dict[str, Any] = {}
            for channel in self._channels:
                values[channel_field(channel, FREQUENCY_FIELD)] = float(
                    self._read_frequency(channel)
                )
                values[channel_field(channel, POWER_FIELD)] = float(
                    self._read_power(channel)
                )
                values[channel_field(channel, OUTPUT_FIELD)] = bool(
                    self._read_output(channel)
                )
            for name, _label, _unit in WINDOW_FIELDS:
                values[name] = float(self._window_value(name))
            return values

    @property
    def identity(self) -> str:
        """The instrument's own name, as its log lines are tagged."""

        return self._identity

    def settings_provenance(self) -> dict[str, object]:
        with self._condition:
            return {
                "device_session_id": self._identity,
                "settings_epoch": self._settings_epoch,
            }

    def tune(self, name: str, value: Any) -> Any:
        try:
            effective = self._resolve_tune(name, value)
        except Exception as error:
            _LOG.info(
                "TUNE REFUSED field=%s value=%r error=%s: %s -- device=%s",
                name,
                value,
                type(error).__name__,
                str(error).replace(chr(10), " "),
                self._identity,
            )
            raise
        _LOG.info(
            "TUNE field=%s value=%r effective=%r device=%s",
            name,
            value,
            effective,
            self._identity,
        )
        return effective

    def _window_value(self, name: str) -> float:
        return {
            "frequency_low_hz": self._frequency_bounds[0],
            "frequency_high_hz": self._frequency_bounds[1],
            "power_low_dbm": self._power_bounds[0],
            "power_high_dbm": self._power_bounds[1],
        }[name]

    def _tune_window(self, selected: str, value: Any) -> float:
        """Move one edge of the bench's window, never a knob under it.

        A change that would strand a channel's CURRENT value outside the
        new window is refused by name: policy may fence a knob in, but
        silently dragging a set output to a new frequency is an output
        change nobody commanded.  Move the knob first, then the fence.
        """

        requested = float(value)
        if not math.isfinite(requested):
            raise ValueError(f"{selected} must be finite")
        with self._condition:
            frequency_low, frequency_high = self._frequency_bounds
            power_low, power_high = self._power_bounds
            window = {
                "frequency_low_hz": (requested, frequency_high),
                "frequency_high_hz": (frequency_low, requested),
                "power_low_dbm": (power_low, power_high),
                "power_high_dbm": (power_low, power_high),
            }
            if selected == "power_low_dbm":
                window[selected] = (requested, power_high)
            elif selected == "power_high_dbm":
                window[selected] = (power_low, requested)
            low, high = window[selected]
            if not low < high:
                raise ValueError(
                    f"{selected}={requested:g} would leave an empty window "
                    f"[{low:g}, {high:g}]"
                )
            frequency_window = selected.startswith("frequency")
            for channel in self._channels:
                current = float(
                    self._read_frequency(channel)
                    if frequency_window
                    else self._read_power(channel)
                )
                if not (low <= current <= high):
                    kind = FREQUENCY_FIELD if frequency_window else POWER_FIELD
                    knob = channel_field(channel, kind)
                    raise ValueError(
                        f"{selected}={requested:g} would strand {knob} at "
                        f"{current:g}; move the knob inside the new window "
                        "first"
                    )
            if frequency_window:
                self._frequency_bounds = (low, high)
            else:
                self._power_bounds = (low, high)
            self._settings_epoch += 1
            self._condition.notify_all()
            return requested

    def _resolve_tune(self, name: str, value: Any) -> Any:
        selected = str(name)
        if any(selected == entry[0] for entry in WINDOW_FIELDS):
            return self._tune_window(selected, value)
        routed = self._routing.get(selected)
        if routed is None:
            offered = ", ".join(sorted(self._routing))
            raise ValueError(
                f"this RF source has no tunable field {selected!r}; it "
                f"offers {offered}"
            )
        channel, kind = routed
        with self._condition:
            if kind == FREQUENCY_FIELD:
                requested = float(value)
                low, high = self._frequency_bounds
                if not (low <= requested <= high):
                    raise ValueError(
                        f"{selected} must lie in [{low:g}, {high:g}] Hz"
                    )
                effective = float(self._write_frequency(channel, requested))
            elif kind == POWER_FIELD:
                requested = float(value)
                low, high = self._power_bounds
                if not (low <= requested <= high):
                    raise ValueError(
                        f"{selected} must lie in [{low:g}, {high:g}] dBm"
                    )
                effective = float(self._write_power(channel, requested))
            else:
                if type(value) is not bool:
                    raise TypeError(f"{selected} takes a bool")
                effective = bool(self._write_output(channel, value))
            self._settings_epoch += 1
            self._condition.notify_all()
            return effective


__all__ = [
    "FREQUENCY_FIELD",
    "OUTPUT_FIELD",
    "POWER_FIELD",
    "WINDOW_FIELDS",
    "RfSource",
    "RfSourceBase",
    "channel_field",
    "snap_to_grid",
]
