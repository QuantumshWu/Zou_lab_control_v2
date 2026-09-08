"""The installed knobs a scan moves, and the two promises every move keeps.

Both engines advance a ``device:`` axis the same way -- a ``tune`` call on
the installed device between fires -- and both owe the bench the same two
things for it.

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

One owner for both, so neither engine can drift from the other about what
a device axis means.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from .plan import DEVICE_PARAM_FAMILY


def device_port_parts(port: str) -> tuple[str, str]:
    """``device:<key>:<field>`` as its installed-device key and field name."""

    text = str(port)
    if not text.startswith(DEVICE_PARAM_FAMILY):
        raise ValueError(f"{port!r} is not a device port")
    key, separator, field = text[len(DEVICE_PARAM_FAMILY):].partition(":")
    if not separator or not key or not field:
        raise ValueError(f"{port!r} names no device field")
    return key, field


def tune_value(device: object, field: str, value: float) -> float:
    """Write a nominal setpoint and return the device's finite readback."""

    effective = device.tune(field, value)
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
        self._pre_run: dict[tuple[str, str], float] = {}

    def move(self, port: str, value: float) -> float:
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
        return tune_value(device, field, float(value))

    @staticmethod
    def _standing(device: object, port: str, field: str) -> float:
        """Where the knob stands, from the device itself, checked to be a
        value the scan could command it back to.

        What is put back is what the operator left, not the value the plan
        happens to start from; and the device's own field bounds are what
        may be commanded, so a standing value outside them is one no
        ``tune`` would ever accept back.
        """

        for entry in device.tunable_fields():
            metadata = entry.metadata
            if metadata.name != field:
                continue
            current = float(entry.current)
            low, high = metadata.minimum, metadata.maximum
            if (low is not None and current < float(low)) or (
                high is not None and current > float(high)
            ):
                raise ValueError(
                    f"{port} stands at {current!r}, outside [{low!r}, {high!r}] "
                    "that may be commanded, so the scan could not put it "
                    "back; move it inside the range first"
                )
            return current
        raise ValueError(f"{port} names no field its device offers")

    def restore(self) -> None:
        """Put every moved knob back where it was found, last moved first.

        Every knob is tried before anything is raised: one refusal must not
        leave the others standing at scan values with nobody told.  Each
        refusal says which knob and which value, because the device's own
        words ("must lie in ...") do not say a scan was putting it back.
        """

        failures: list[BaseException] = []
        for (key, field), value in reversed(self._pre_run.items()):
            try:
                tune_value(self._tunables[key], field, value)
            except BaseException as error:
                failure = RuntimeError(
                    f"device field {field!r} of {key!r} was not put back to "
                    f"its pre-run value {value!r}: {error}"
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


def release_after_scan(
    steps: tuple[tuple[str, object], ...], error: BaseException | None
) -> None:
    """Run every named cleanup step, and tell about all of them.

    On the way out of a failed scan the original error stays the error:
    what cleanup could not do is attached to it as notes.  On the way out
    of a successful one, a cleanup failure is the run's failure, raised
    after every step was still attempted.
    """

    failures: list[tuple[str, BaseException]] = []
    for name, step in steps:
        try:
            step()
        except BaseException as failure:
            failures.append((name, failure))
    if error is not None:
        for name, failure in failures:
            error.add_note(
                f"{name} also reported: {type(failure).__name__}: {failure}"
            )
        return
    if len(failures) == 1:
        raise failures[0][1]
    if failures:
        raise BaseExceptionGroup(
            "ending the scan failed", [failure for _name, failure in failures]
        )


__all__ = [
    "ScanDeviceKnobs",
    "device_port_parts",
    "release_after_scan",
    "tune_value",
]
