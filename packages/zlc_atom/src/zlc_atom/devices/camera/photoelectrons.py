"""Reading a camera in photoelectrons: ``(count - offset) * electrons_per_count``.

A count is what the sensor's converter says.  A photoelectron is how many
electrons were collected -- a plain number, with nothing to attach to it --
so this file names no unit and invents none.  What the conversion buys is a
frame whose numbers a physicist reads directly; what it costs is float32.

The two numbers are the camera's, written down on the DEVICE beside its ROI
and its exposure, because they are a fact about that sensor rather than about
any one run.  Photoelectrons are requested by default.  A camera that states
no complete conversion exposes the switch as unavailable and resolves that
request effectively to raw counts; the Dataset unit and run record say what
actually happened.  Half a conversion remains an invalid device
configuration rather than a fallback.
"""

from __future__ import annotations

from collections.abc import Mapping
import math

from zlc_atom.authoring import AuthoringField


#: The authored switch every camera-fed node uses for this choice.
PHOTOELECTRONS = "photoelectrons"


def photoelectron_switch(
    *,
    enabled_when: tuple[str, tuple[object, ...]] | None = None,
) -> AuthoringField:
    """Read this camera in photoelectrons instead of counts: on or off.

    On by default.  A capable camera converts through its own stated offset
    and scale.  For a camera with no conversion, the disabled control shows
    the reason while its effective value is False; the authored value is kept
    so selecting a capable camera restores it.  Direct requests use the same
    raw-count fallback and record the effective choice.
    """

    return AuthoringField(
        PHOTOELECTRONS,
        "bool",
        "Photoelectrons",
        True,
        enabled_when=enabled_when,
    )


def stated_conversion(
    offset_counts: float | None,
    electrons_per_count: float | None,
    *,
    camera: str,
) -> tuple[float, float] | None:
    """One camera's configured conversion: both numbers, or none at all.

    Half a conversion is not a weaker one, it is a wrong one, so a config
    that states an offset without a scale is refused where it is written
    rather than where a frame is read.
    """

    if offset_counts is None and electrons_per_count is None:
        return None
    if offset_counts is None or electrons_per_count is None:
        raise ValueError(
            f"{camera} needs both an offset and electrons per count, or neither"
        )
    offset = float(offset_counts)
    electrons = float(electrons_per_count)
    if not math.isfinite(offset) or not math.isfinite(electrons) or electrons <= 0.0:
        raise ValueError(
            f"{camera} electrons per count must be finite and positive, and "
            "its offset finite"
        )
    return (offset, electrons)


def photoelectron_conversion_of(camera: object) -> tuple[float, float] | None:
    """What one of this camera's counts is worth, or None if it does not say."""

    if camera is None:
        return None
    conversion = getattr(camera, "photoelectron_conversion", None)
    if conversion is None:
        return None
    offset, electrons_per_count = conversion
    return (float(offset), float(electrons_per_count))


def resolve_photoelectron_availability(
    devices: Mapping[str, object],
) -> dict[str, str]:
    """The descriptor hook: why this draft's camera cannot be read that way.

    Empty means it can.  Answered from the device's own configuration, so it
    is knowable with the camera shut -- which is what lets a form disable the
    switch and show its effective raw-count fallback before anything is armed.
    """

    if photoelectron_conversion_of(devices.get("camera")) is not None:
        return {}
    return {
        PHOTOELECTRONS: (
            "the selected camera states no photoelectron conversion: set its "
            "offset and electrons per count in Device Manager"
        )
    }


__all__ = [
    "PHOTOELECTRONS",
    "photoelectron_conversion_of",
    "photoelectron_switch",
    "resolve_photoelectron_availability",
    "stated_conversion",
]
