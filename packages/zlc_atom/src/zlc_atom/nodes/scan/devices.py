"""The installed knobs a scan moves, and the two promises every move keeps.

A ``device:`` axis advances by a ``tune`` call on the installed device
between fires, and the scan owes the bench two things for it.

THE SETPOINT AND READBACK ARE DIFFERENT FACTS. The scan coordinate is the
nominal setpoint; ``tune`` returns the instrument's actual numeric readback
for the run record. Rounding or cutoff does not make a successful device
command fail merely because these values differ.

THE BENCH IS HANDED BACK AS IT WAS FOUND.  A scan ends -- complete,
stopped or failed -- with every knob it moved back at its pre-run value,
read from the device before the first move and written back through the
same ``tune`` path. A device refusal is reported, not inferred from numeric
equality. A synthesizer left standing at the last scan
point was what the operator found after every scan, and nothing on the
bench said so.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from zlc_atom.authoring import read_tunable_in_unit, refresh_tunable_fields, tune_in_unit

from .plan import device_port_parts


def tune_value(device: object, field: str, value: float, unit: str = "") -> float:
    """Write a nominal setpoint and return the device's finite readback."""

    effective = tune_in_unit(device, field, value, unit)
    if isinstance(effective, bool):
        raise TypeError("device tune must return its effective numeric value")
    try:
        actual = float(effective)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "device tune must return its effective numeric value"
        ) from error
    if not math.isfinite(actual):
        raise ValueError("device tune returned a non-finite effective value")
    return actual


class ScanDeviceKnobs:
    """The knobs one scan moves: remembered before the first move, put back
    at the end."""

    def __init__(self, tunables: Mapping[str, object] | None) -> None:
        self._tunables = dict(tunables or {})
        #: (device key, field) -> the value the device reported before this
        #: scan first moved it, in the order the fields were first moved.
        self._pre_run: dict[tuple[str, str], tuple[float, str]] = {}

    def move(self, port: str, value: float, unit: str = "") -> float:
        """Set one knob to a scan coordinate, remembering where it stood.

        The first move of a knob is when the promise to put it back is
        made, so that is when it is checked: a knob standing outside the
        range that may be commanded -- an instrument an operator left
        below the bench's policy window -- could never be put back, and
        the scan says so before it moves anything, not after it has taken
        the data.
        """

        key, field = device_port_parts(port)
        device = self._tunables.get(key)
        if device is None:
            raise ValueError(
                f"this bench offers no tunable device {key!r} for port {port!r}"
            )
        if (key, field) not in self._pre_run:
            self._pre_run[(key, field)] = self._standing(device, port, field)
        return tune_value(device, field, float(value), unit)

    @staticmethod
    def _standing(device: object, port: str, field: str) -> tuple[float, str]:
        """Where the knob stands, from the device itself, checked to be a
        value the scan could command it back to.

        What is put back is what the operator left, not the value the plan
        happens to start from; and the device's own field bounds are what
        may be commanded, so a standing value outside them is one no
        ``tune`` would ever accept back.
        """

        entry = read_tunable_in_unit(device, field)
        if entry.current is None:
            refresh_tunable_fields(device)
            entry = read_tunable_in_unit(device, field)
        metadata = entry.metadata
        current = float(entry.current)
        low, high = metadata.minimum, metadata.maximum
        if (low is not None and current < float(low)) or (
            high is not None and current > float(high)
        ):
            raise ValueError(
                f"{port} stands at {current!r} {metadata.unit}, outside [{low!r}, {high!r}] "
                "that may be commanded, so the scan could not put it "
                "back; move it inside the range first"
            )
        return current, metadata.unit

    def restore(self) -> None:
        """Put every moved knob back where it was found, last moved first.

        Every knob is tried before anything is raised: one refusal must not
        leave the others standing at scan values with nobody told.  Each
        refusal says which knob and which value, because the device's own
        words ("must lie in ...") do not say a scan was putting it back.
        """

        failures: list[BaseException] = []
        for (key, field), (value, unit) in reversed(self._pre_run.items()):
            try:
                tune_value(self._tunables[key], field, value, unit)
            except BaseException as error:
                failure = RuntimeError(
                    f"device field {field!r} of {key!r} was not put back to "
                    f"its pre-run value {value!r} {unit}: {error}"
                )
                failure.__cause__ = error
                failures.append(failure)
        self._pre_run.clear()
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup(
                "restoring the scanned device fields failed", failures
            )


__all__ = [
    "ScanDeviceKnobs",
    "tune_value",
]
