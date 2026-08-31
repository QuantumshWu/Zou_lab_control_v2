"""The device-independent RF source surface every implementation answers.

An RF source is, to this system, its TUNABLE SURFACE: a frequency, a power,
and an output switch, each read and written through the same duck-typed
quartet every tunable device speaks (``tunable_fields`` / ``tune`` /
``tunable_values`` / ``settings_provenance``).  A scan axis, the generic
control panel and the device-axis executor all consume exactly that quartet,
so the capability Protocol IS the quartet -- there is no second, RF-only
vocabulary for a consumer to learn.

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

import math
import threading
from typing import Any, Mapping, Protocol, runtime_checkable

from zlc_atom.authoring import AuthoringField, TunableField

FREQUENCY_FIELD = "frequency_hz"
POWER_FIELD = "power_dbm"
OUTPUT_FIELD = "output_enabled"


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


class RfSourceBase:
    """The shared half of every RF driver: fields, bounds, epochs, locking.

    A concrete driver supplies the transport verbs (``_write_frequency`` and
    friends), each of which performs the hardware write AND returns the
    instrument's own read-back.  Everything the consumers see -- the tunable
    quartet -- lives here once.
    """

    def __init__(
        self,
        *,
        identity: str,
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
        self._identity = str(identity)
        self._frequency_bounds = (float(frequency_low_hz), float(frequency_high_hz))
        self._power_bounds = (float(power_low_dbm), float(power_high_dbm))
        self._condition = threading.Condition()
        self._settings_epoch = 0

    # ------------------------------------------------------- transport verbs
    def _write_frequency(self, value_hz: float) -> float:
        raise NotImplementedError

    def _write_power(self, value_dbm: float) -> float:
        raise NotImplementedError

    def _write_output(self, enabled: bool) -> bool:
        raise NotImplementedError

    def _read_frequency(self) -> float:
        raise NotImplementedError

    def _read_power(self) -> float:
        raise NotImplementedError

    def _read_output(self) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    # -------------------------------------------------------------- contract
    def tunable_fields(self) -> tuple[TunableField, ...]:
        frequency_low, frequency_high = self._frequency_bounds
        power_low, power_high = self._power_bounds
        with self._condition:
            frequency = float(self._read_frequency())
            power = float(self._read_power())
            output = bool(self._read_output())
        return (
            TunableField(
                metadata=AuthoringField(
                    FREQUENCY_FIELD,
                    "float",
                    "Frequency (Hz)",
                    frequency_low,
                    minimum=frequency_low,
                    maximum=frequency_high,
                    unit="Hz",
                ),
                current=frequency,
                live_write=True,
                dependency_group=(FREQUENCY_FIELD,),
            ),
            TunableField(
                metadata=AuthoringField(
                    POWER_FIELD,
                    "float",
                    "Power (dBm)",
                    power_low,
                    minimum=power_low,
                    maximum=power_high,
                    unit="dBm",
                ),
                current=power,
                live_write=True,
                dependency_group=(POWER_FIELD,),
            ),
            # No bounds on purpose: a bool is a switch, not a scan axis, and
            # scan_ports_for_devices admits only bounded fields -- so the
            # output toggle appears on the control panel and never in the
            # add-axis combo.
            TunableField(
                metadata=AuthoringField(
                    OUTPUT_FIELD,
                    "bool",
                    "Output enabled",
                    False,
                ),
                current=output,
                live_write=True,
                dependency_group=(OUTPUT_FIELD,),
            ),
        )

    def tunable_values(self) -> dict[str, Any]:
        with self._condition:
            return {
                FREQUENCY_FIELD: float(self._read_frequency()),
                POWER_FIELD: float(self._read_power()),
                OUTPUT_FIELD: bool(self._read_output()),
            }

    def settings_provenance(self) -> dict[str, object]:
        with self._condition:
            return {
                "device_session_id": self._identity,
                "settings_epoch": self._settings_epoch,
            }

    def tune(self, name: str, value: Any) -> Any:
        selected = str(name)
        with self._condition:
            if selected == FREQUENCY_FIELD:
                requested = float(value)
                low, high = self._frequency_bounds
                if not (low <= requested <= high):
                    raise ValueError(
                        f"{FREQUENCY_FIELD} must lie in [{low:g}, {high:g}] Hz"
                    )
                effective = float(self._write_frequency(requested))
            elif selected == POWER_FIELD:
                requested = float(value)
                low, high = self._power_bounds
                if not (low <= requested <= high):
                    raise ValueError(
                        f"{POWER_FIELD} must lie in [{low:g}, {high:g}] dBm"
                    )
                effective = float(self._write_power(requested))
            elif selected == OUTPUT_FIELD:
                if type(value) is not bool:
                    raise TypeError(f"{OUTPUT_FIELD} takes a bool")
                effective = bool(self._write_output(value))
            else:
                raise ValueError(
                    f"this RF source has no tunable field {selected!r}; it "
                    f"offers {FREQUENCY_FIELD!r}, {POWER_FIELD!r} and "
                    f"{OUTPUT_FIELD!r}"
                )
            self._settings_epoch += 1
            self._condition.notify_all()
            return effective


__all__ = [
    "FREQUENCY_FIELD",
    "OUTPUT_FIELD",
    "POWER_FIELD",
    "RfSource",
    "RfSourceBase",
    "snap_to_grid",
]
