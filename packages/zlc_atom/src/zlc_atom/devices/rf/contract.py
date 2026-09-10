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
source keeps the bare field names (``frequency``); a multi-channel one
prefixes them with its own channel names (``ch1_frequency``), so the
add-axis combo and the control panel show every knob of the one instrument
under its one card.

TWO FENCES, ONE RANGE.  A knob is bounded by the instrument's own limits,
read from the device when the connection opens, and by an optional bench
policy window, authored at Init and adjustable on the control panel.  What
may be COMMANDED -- by the panel, by a notebook, by a scan -- is the
tighter of the two on each side, so a scan range exists whenever the
instrument has a range, and a missing policy edge never forbids a sweep.
The instrument's limits are exposed beside the effective bounds
(``TunableField.device_limits``) so the panel can show which fence bites.

What varies between instruments is the transport underneath (SCPI text over
VISA for a bench generator, a vendor DLL for a Lab Brick) and the value grid
the hardware quantizes to.  Both live in the concrete drivers; the shared
plumbing here owns the rules that must not fork per driver:

* a write is SET-THEN-READ-BACK, and ``tune`` returns what the instrument
  itself reports, never what was asked;
* a value the instrument would silently round is REFUSED before it is
  written, naming the grid -- a scan coordinate must mean exactly what its
  dataset column says (the same law the pulse DAC axes obey);
* ``settings_epoch`` compares the actual write result with the session's
  latest known value. It requires no extra device query or saved history.
  Explicit Refresh adopts external front-panel changes.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any, Mapping, Protocol, runtime_checkable
from uuid import uuid4

from zlc_atom.authoring import AuthoringField, TunableField

FREQUENCY_FIELD = "frequency"
POWER_FIELD = "power"
OUTPUT_FIELD = "output_enabled"

#: The bench's safety window is a CONTROL knob, not an apparatus fact: the
#: operator adjusts it on the control panel (plain Apply, never live) or
#: through the same ``tune`` API, and nothing else may scan it.  It is
#: deliberately unbounded and non-live so scan_ports_for_devices never
#: offers it as an axis. (runtime name, label, unit, fixed-unit Init key).
WINDOW_FIELDS = (
    ("frequency_low", "Frequency low", "Hz", "frequency_low_hz"),
    ("frequency_high", "Frequency high", "Hz", "frequency_high_hz"),
    ("power_low", "Power low", "dBm", "power_low_dbm"),
    ("power_high", "Power high", "dBm", "power_high_dbm"),
)

#: The same optional policy fields are authored at Init and exposed again by
#: Device Control.  ``None`` has one meaning everywhere: this bench has not
#: imposed that edge.  Keeping the declarations here prevents real and
#: virtual RF families from inventing different defaults or units.
WINDOW_AUTHORING_FIELDS = tuple(
    AuthoringField(config_name, "float", label, None, unit=unit)
    for _name, label, unit, config_name in WINDOW_FIELDS
)


def validate_window_values(values: Mapping[str, object]) -> None:
    """Validate the optional RF policy edges in any device authoring schema."""

    for prefix in ("frequency", "power"):
        low = values[f"{prefix}_low_{'hz' if prefix == 'frequency' else 'dbm'}"]
        high = values[f"{prefix}_high_{'hz' if prefix == 'frequency' else 'dbm'}"]
        if low is not None and high is not None and float(low) >= float(high):
            raise ValueError(
                f"{prefix} low must be below {prefix} high when both are set"
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

    Construction is two steps, because they are two different facts.
    ``__init__`` takes the AUTHORED half -- channels and the bench's policy
    window -- and touches no transport, so a window that cannot be honoured
    is refused before any instrument is opened.  Once the driver's transport
    is open it calls ``_attach`` with the instrument's own name, which reads
    the hardware limits and records the session; a driver whose attach
    fails closes its transport and raises, so nothing it opened is left
    without an owner.
    """

    def __init__(
        self,
        *,
        channels: tuple[str, ...] = ("",),
        frequency_low_hz: float | None,
        frequency_high_hz: float | None,
        power_low_dbm: float | None,
        power_high_dbm: float | None,
    ) -> None:
        frequency_bounds = self._policy_window(
            frequency_low_hz,
            frequency_high_hz,
            name="frequency",
        )
        power_bounds = self._policy_window(
            power_low_dbm,
            power_high_dbm,
            name="power",
        )
        names = tuple(str(channel) for channel in channels)
        if not names or len(set(names)) != len(names):
            raise ValueError("rf channels must be non-empty and unique")
        if len(names) > 1 and any(not name for name in names):
            raise ValueError("a multi-channel source names every channel")
        self._identity = ""
        self._channels = names
        self._frequency_bounds = frequency_bounds
        self._power_bounds = power_bounds
        #: channel -> (frequency limits, power limits), the instrument's own,
        #: read once by ``_attach``.
        self._device_limits: dict[
            str, tuple[tuple[float, float], tuple[float, float] | None]
        ] = {}
        self._condition = threading.Condition()
        self._settings_epoch = 0
        # One latest reading per knob, never a settings history. Metadata
        # projection and a write's epoch must not query the whole instrument.
        self._current_values: dict[str, Any] = {}
        # Minted per connection, never derived from the instrument: two
        # sessions over the same brick are two histories of settings, and a
        # risk acceptance bound to one must not survive into the other.
        self._device_session_id = uuid4().hex
        #: field name -> (channel, kind); the ONE table tune() resolves by,
        #: so a field's spelling cannot drift from its routing.
        self._routing: dict[str, tuple[str, str]] = {}
        for channel in names:
            for kind in (FREQUENCY_FIELD, POWER_FIELD, OUTPUT_FIELD):
                self._routing[channel_field(channel, kind)] = (channel, kind)
        # OPENING IS READ-ONLY.  Connecting to an instrument is an act of
        # observation: whatever it was doing when the operator plugged in
        # is the experiment's state, and the app has no mandate to edit it
        # for having been asked to look.  The bench window bounds what may
        # be COMMANDED -- ``tune`` refuses outside it, and tightening an
        # edge past a live knob is refused by name rather than dragging the
        # knob -- so opening had a second, silent answer to the question
        # the rest of this class answers by refusing.  A knob idling
        # outside policy is now a reading the panel shows, which is the
        # only way the operator can find out.

    def _attach(self, identity: str) -> None:
        """Take charge of the instrument the transport now reaches.

        The instrument's own limits are facts about its model and firmware,
        learned once per connection and answered from memory afterwards;
        every bound check -- the panel's, a notebook's, a scan's -- reads
        the same numbers without a round trip.  A bench window that leaves
        nothing of the instrument's range is refused here, by name, instead
        of surfacing later as a knob with an empty scan range.
        """

        limits: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {}
        for channel in self._channels:
            frequency_limits = self._instrument_limits(
                self._read_frequency_limits(channel),
                name=channel_field(channel, FREQUENCY_FIELD),
                unit="Hz",
            )
            power_limits = self._instrument_limits(
                self._read_power_limits(channel),
                name=channel_field(channel, POWER_FIELD),
                unit="dBm",
            )
            self._effective_range(
                self._frequency_bounds,
                frequency_limits,
                name=channel_field(channel, FREQUENCY_FIELD),
                unit="Hz",
            )
            self._effective_range(
                self._power_bounds,
                power_limits,
                name=channel_field(channel, POWER_FIELD),
                unit="dBm",
            )
            limits[channel] = (frequency_limits, power_limits)
        self._identity = str(identity)
        self._device_limits = limits
        # Construction already read the driver's channel facts with its
        # limits. Seed current values without repeating an explicit refresh.
        RfSourceBase.tunable_values(self)

    @staticmethod
    def _optional_edge(value: object, *, name: str) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool):
            raise TypeError(f"{name} must be a finite number or None")
        edge = float(value)
        if not math.isfinite(edge):
            raise ValueError(f"{name} must be finite or None")
        return edge

    @classmethod
    def _policy_window(
        cls,
        low: object,
        high: object,
        *,
        name: str,
    ) -> tuple[float | None, float | None]:
        lower = cls._optional_edge(low, name=f"{name} low")
        upper = cls._optional_edge(high, name=f"{name} high")
        if lower is not None and upper is not None and lower >= upper:
            raise ValueError(f"{name} bounds must be ordered when both are set")
        return lower, upper

    @staticmethod
    def _instrument_limits(
        limits: tuple[float, float], *, name: str, unit: str
    ) -> tuple[float, float]:
        """The instrument's answer, checked to be a range at all."""

        low, high = (float(edge) for edge in limits)
        if not (math.isfinite(low) and math.isfinite(high)) or low > high:
            raise RuntimeError(
                f"the instrument reports {name} limits [{low!r}, {high!r}] "
                f"{unit}, which is not a range"
            )
        return low, high

    @staticmethod
    def _effective_range(
        window: tuple[float | None, float | None],
        limits: tuple[float, float],
        *,
        name: str,
        unit: str,
    ) -> tuple[float, float]:
        """What may be commanded: the tighter of window and limits, per side.

        The bench's window narrows what the instrument allows; it never has
        to exist for the instrument's range to, and it cannot widen it.  A
        window that excludes the whole instrument range is a contradiction,
        not a narrow fence, and is refused by name.
        """

        low, high = window
        limit_low, limit_high = limits
        lower = limit_low if low is None else max(float(low), limit_low)
        upper = limit_high if high is None else min(float(high), limit_high)
        if lower > upper:
            raise ValueError(
                f"{name}: the bench window [{low!r}, {high!r}] {unit} leaves "
                f"nothing of the instrument's own range "
                f"[{limit_low!r}, {limit_high!r}] {unit}"
            )
        return lower, upper

    def _frequency_range(self, channel: str) -> tuple[float, float]:
        return self._effective_range(
            self._frequency_bounds,
            self._device_limits[channel][0],
            name=channel_field(channel, FREQUENCY_FIELD),
            unit="Hz",
        )

    def _power_range(self, channel: str) -> tuple[float | None, float | None]:
        if self._device_limits[channel][1] is None:
            return self._power_bounds
        return self._effective_range(
            self._power_bounds,
            self._device_limits[channel][1],
            name=channel_field(channel, POWER_FIELD),
            unit="dBm",
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

    def _read_frequency_limits(self, channel: str) -> tuple[float, float]:
        """The instrument's own frequency range for this channel, in hertz."""

        raise NotImplementedError

    def _read_power_limits(self, channel: str) -> tuple[float, float]:
        """The instrument's own power range for this channel, in dBm."""

        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    # -------------------------------------------------------------- contract
    def _channel_label(self, channel: str) -> str:
        return f"{channel.upper()} · " if channel else ""

    def tunable_fields(self) -> tuple[TunableField, ...]:
        fields: list[TunableField] = []
        with self._condition:
            for channel in self._channels:
                label = self._channel_label(channel)
                frequency = self._current_values.get(channel_field(channel, FREQUENCY_FIELD))
                frequency_limits, power_limits = self._device_limits[channel]
                frequency_range = self._frequency_range(channel)
                power_range = self._power_range(channel)
                fields.append(
                    TunableField(
                        metadata=AuthoringField(
                            channel_field(channel, FREQUENCY_FIELD),
                            "float",
                            f"{label}Frequency",
                            # A live knob has no draft default: what it is
                            # right now is ``current``, and repeating the
                            # reading here would make a knob idling outside
                            # bench policy unstateable -- the field would
                            # refuse its own instrument.
                            None,
                            minimum=frequency_range[0],
                            maximum=frequency_range[1],
                            unit="Hz",
                        ),
                        current=frequency,
                        live_write=True,
                        dependency_group=(
                            channel_field(channel, FREQUENCY_FIELD),
                        ),
                        device_limits=frequency_limits,
                    )
                )
                power = self._current_values.get(channel_field(channel, POWER_FIELD))
                fields.append(
                    TunableField(
                        metadata=AuthoringField(
                            channel_field(channel, POWER_FIELD),
                            "float",
                            f"{label}Power",
                            None,
                            minimum=power_range[0],
                            maximum=power_range[1],
                            unit="dBm",
                        ),
                        current=power,
                        live_write=True,
                        dependency_group=(channel_field(channel, POWER_FIELD),),
                        device_limits=power_limits,
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
                        current=self._current_values.get(channel_field(channel, OUTPUT_FIELD)),
                        live_write=True,
                        dependency_group=(channel_field(channel, OUTPUT_FIELD),),
                    )
                )
            for name, label, unit, _config_name in WINDOW_FIELDS:
                current = self._window_value(name)
                fields.append(
                    TunableField(
                        metadata=AuthoringField(
                            name, "float", label, None, unit=unit
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
            for name, _label, _unit, _config_name in WINDOW_FIELDS:
                values[name] = self._window_value(name)
            self._current_values.update(values)
            return values

    def refresh_tunable_fields(self) -> tuple[TunableField, ...]:
        """Explicit hardware refresh followed by the ordinary projection."""

        with self._condition:
            self.tunable_values()
            return self.tunable_fields()

    @property
    def identity(self) -> str:
        """The instrument's own name, as its log lines are tagged."""

        return self._identity

    def settings_provenance(self) -> dict[str, object]:
        with self._condition:
            return {
                "device_session_id": self._device_session_id,
                "settings_epoch": self._settings_epoch,
            }

    def tune(self, name: str, value: Any) -> Any:
        return self._tune_logged(name, value)

    def _tune_logged(self, name: str, value: Any, *, unit: str = "") -> Any:
        try:
            effective = self._resolve_tune(name, value, unit=unit)
        except Exception as error:
            _LOG.info(
                "TUNE REFUSED field=%s value=%r unit=%s error=%s: %s -- device=%s",
                name,
                value,
                unit or "canonical",
                type(error).__name__,
                str(error).replace(chr(10), " "),
                self._identity,
            )
            raise
        _LOG.info(
            "TUNE field=%s value=%r effective=%r unit=%s device=%s",
            name,
            value,
            effective,
            unit or "canonical",
            self._identity,
        )
        return effective

    def _window_value(self, name: str) -> float | None:
        return {
            "frequency_low": self._frequency_bounds[0],
            "frequency_high": self._frequency_bounds[1],
            "power_low": self._power_bounds[0],
            "power_high": self._power_bounds[1],
        }[name]

    def _tune_window(self, selected: str, value: Any) -> float | None:
        """Move one edge of the bench's window, never a knob under it.

        A change that would strand a channel's CURRENT value outside the
        new window is refused by name: policy may fence a knob in, but
        silently dragging a set output to a new frequency is an output
        change nobody commanded.  Move the knob first, then the fence.  An
        edge that would leave nothing of the instrument's own range is
        refused the same way, because there is no knob position that
        could ever satisfy it.
        """

        requested = self._optional_edge(value, name=selected)
        with self._condition:
            frequency_low, frequency_high = self._frequency_bounds
            power_low, power_high = self._power_bounds
            window = {
                "frequency_low": (requested, frequency_high),
                "frequency_high": (frequency_low, requested),
                "power_low": (requested, power_high),
                "power_high": (power_low, requested),
            }
            low, high = window[selected]
            if low is not None and high is not None and low >= high:
                raise ValueError(
                    f"{selected}={requested!r} would leave an empty window "
                    f"[{low!r}, {high!r}]"
                )
            frequency_window = selected.startswith("frequency")
            unit = "Hz" if frequency_window else "dBm"
            kind = FREQUENCY_FIELD if frequency_window else POWER_FIELD
            for channel in self._channels:
                knob = channel_field(channel, kind)
                limits = self._device_limits[channel][0 if frequency_window else 1]
                if limits is not None:
                    self._effective_range((low, high), limits, name=knob, unit=unit)
                current = float(
                    self._read_frequency(channel)
                    if frequency_window
                    else self._read_power(channel)
                )
                below = low is not None and current < low
                above = high is not None and current > high
                if below or above:
                    raise ValueError(
                        f"{selected}={requested!r} would strand {knob} at "
                        f"{current:g}; move the knob inside the new window "
                        "first"
                    )
            previous = self._window_value(selected)
            if frequency_window:
                self._frequency_bounds = (low, high)
            else:
                self._power_bounds = (low, high)
            if requested != previous:
                self._settings_epoch += 1
                self._condition.notify_all()
            return requested

    def _resolve_tune(self, name: str, value: Any, *, unit: str = "") -> Any:
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
            # Compare the actual result with this session's latest reading;
            # no extra before-query merely to count an epoch.
            before = self._current_values.get(selected)
            if kind == FREQUENCY_FIELD:
                requested = float(value)
                low, high = self._frequency_range(channel)
                if not low <= requested <= high:
                    raise ValueError(
                        f"{selected} must lie in [{low!r}, {high!r}] Hz"
                    )
                self._current_values.pop(selected, None)
                effective: Any = float(self._write_frequency(channel, requested))
            elif kind == POWER_FIELD:
                requested = float(value)
                reader = getattr(self, "read_tunable_in_unit", None)
                reading = reader(selected, unit or "dBm") if callable(reader) else None
                low, high = (
                    (reading.metadata.minimum, reading.metadata.maximum)
                    if reading is not None else self._power_range(channel)
                )
                if (low is not None and requested < low) or (high is not None and requested > high):
                    raise ValueError(
                        f"{selected} must lie in [{low!r}, {high!r}] {unit or 'dBm'}"
                    )
                self._current_values.pop(selected, None)
                effective = float(
                    self._write_power_in_unit(channel, requested, unit)
                    if unit else self._write_power(channel, requested)
                )
            else:
                if type(value) is not bool:
                    raise TypeError(f"{selected} takes a bool")
                self._current_values.pop(selected, None)
                effective = bool(self._write_output(channel, value))
            canonical = (
                self.convert_tunable_value(selected, effective, unit, "dBm")
                if kind == POWER_FIELD and unit else effective
            )
            self._current_values[selected] = canonical
            if canonical != before:
                self._settings_epoch += 1
                self._condition.notify_all()
            return effective


__all__ = [
    "FREQUENCY_FIELD",
    "OUTPUT_FIELD",
    "POWER_FIELD",
    "WINDOW_FIELDS",
    "WINDOW_AUTHORING_FIELDS",
    "validate_window_values",
    "RfSource",
    "RfSourceBase",
    "channel_field",
    "snap_to_grid",
]
