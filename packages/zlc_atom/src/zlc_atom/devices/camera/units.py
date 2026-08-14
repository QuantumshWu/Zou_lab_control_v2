"""What the numbers in a frame ARE: the sensor's counts, or photoelectrons.

The conversion between them is the CAMERA's -- an offset and what one count
is worth -- and it is configured on the device, beside its ROI and exposure,
because it is a fact about that sensor rather than about any one run.  Nobody
downstream may invent it: a camera that states none can only publish counts.

Which of the two a run publishes is still the run's choice, because the
conversion costs float32 frames.  So the choice is offered as the two units
this bench understands, with the camera's own numbers written into the option
that uses them -- and, when the bound camera states nothing, with the reason
it cannot be picked, rather than an option that quietly disappears.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
import math

from zlc_atom.authoring import AuthoringChoice, AuthoringField


#: The authored field name every camera-fed node uses for this choice.
FRAME_UNITS = "frame_units"


class FrameUnit(str, Enum):
    """The unit of every number in a published frame."""

    COUNTS = "counts"
    PHOTOELECTRONS = "photoelectrons"


def frame_unit_field(
    *,
    enabled_when: tuple[str, tuple[object, ...]] | None = None,
) -> AuthoringField:
    """The unit choice as it stands before any camera has been bound."""

    return AuthoringField(
        FRAME_UNITS,
        "choice",
        "Frame units",
        FrameUnit.COUNTS.value,
        choices=frame_unit_choices(None),
        enabled_when=enabled_when,
    )


def frame_unit_choices(camera: object) -> tuple[AuthoringChoice, ...]:
    """The two units, as the bound camera's configuration answers for them."""

    conversion = photoelectron_conversion_of(camera)
    if conversion is None:
        label = "Photoelectrons (camera states no conversion)"
        reason = (
            "the selected camera states no photoelectron conversion: set its "
            "offset and electrons per count in Device Manager, or publish counts"
        )
    else:
        offset, electrons_per_count = conversion
        label = f"Photoelectrons ({electrons_per_count:g} e-/count, offset {offset:g})"
        reason = ""
    return (
        AuthoringChoice(FrameUnit.COUNTS.value, "Counts (sensor ADU)"),
        AuthoringChoice(FrameUnit.PHOTOELECTRONS.value, label, unavailable_reason=reason),
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


def resolve_frame_unit_choices(
    devices: Mapping[str, object],
) -> dict[str, tuple[AuthoringChoice, ...]]:
    """The descriptor hook: this draft's unit options, given its bound camera."""

    return {FRAME_UNITS: frame_unit_choices(devices.get("camera"))}


__all__ = [
    "FRAME_UNITS",
    "FrameUnit",
    "frame_unit_choices",
    "frame_unit_field",
    "photoelectron_conversion_of",
    "resolve_frame_unit_choices",
    "stated_conversion",
]
