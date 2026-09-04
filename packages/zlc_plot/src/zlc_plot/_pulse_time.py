"""Time-coordinate conversion shared by pulse rendering and interaction."""

from __future__ import annotations

import math
from numbers import Real

from zlc_data.units import UnitError, prefix_for, resolve_unit

from .primitives import PulseTimelineData


_MINIMUM_SOURCE_SPAN = 1e-12


def _time_unit(unit: str) -> object | None:
    """One pulse time unit, or None for a source this cannot rescale.

    This module used to keep its own four-row table AND its own spelling
    rules, which had drifted the opposite way from the model's: the pulse
    canonicalises to ``µs``, and this canonicalised to ``us``.  Two answers to
    "what is this axis in" is one more than a timeline can have.
    """

    if not isinstance(unit, str):
        raise TypeError("pulse time unit must be text")
    try:
        resolved = resolve_unit(unit.strip())
    except UnitError:
        return None
    return resolved if resolved.dimension == "time" else None


def pulse_content_bounds(payload: PulseTimelineData) -> tuple[float, float]:
    """Return the complete authored pulse bounds in the source time unit."""

    if not isinstance(payload, PulseTimelineData):
        raise TypeError("payload must be PulseTimelineData")
    starts = [0.0]
    stops = [_MINIMUM_SOURCE_SPAN]
    starts.extend(block.start for block in payload.blocks)
    stops.extend(block.stop for block in payload.blocks)
    starts.extend(region.start for region in payload.scan_regions)
    stops.extend(region.stop for region in payload.scan_regions)
    for trace in payload.analog_traces:
        if trace.starts:
            starts.append(trace.starts[0])
            stops.append(trace.starts[-1])
    starts.extend(marker.start for marker in payload.loop_markers)
    stops.extend(marker.stop for marker in payload.loop_markers)
    starts.extend(segment.start for segment in payload.scan_dac_segments)
    stops.extend(segment.stop for segment in payload.scan_dac_segments)
    if payload.total_duration is not None:
        stops.append(payload.total_duration)
    start = min(starts)
    stop = max(stops)
    if stop - start < _MINIMUM_SOURCE_SPAN:
        stop = start + _MINIMUM_SOURCE_SPAN
    return float(start), float(stop)


def pulse_time_scale(
    payload: PulseTimelineData,
    display_unit: str | None = None,
    *,
    source_span: float | None = None,
) -> tuple[float, str]:
    """Return ``source value * factor -> displayed value`` and its unit."""

    if not isinstance(payload, PulseTimelineData):
        raise TypeError("payload must be PulseTimelineData")
    source = _time_unit(payload.time_unit)
    if source is None:
        if display_unit is None or _time_unit(display_unit) is None:
            return 1.0, payload.time_unit
        raise ValueError(
            f"cannot convert pulse source unit {payload.time_unit!r} to {display_unit!r}"
        )
    if display_unit is None:
        if source_span is None:
            start, stop = pulse_content_bounds(payload)
            span = stop - start
        else:
            if isinstance(source_span, bool) or not isinstance(source_span, Real):
                raise TypeError("pulse source span must be a real number")
            span = float(source_span)
        if not math.isfinite(span) or span <= 0.0:
            raise ValueError("pulse source span must be positive and finite")
        # The same rule that sizes every other number in this project: put
        # the value between one and a thousand.  This used to be a fourth
        # hand-written ladder, so a timeline could disagree with the editor
        # about which unit a duration reads best in.
        seconds = abs(span) * source.scale
        target = resolve_unit(f"{prefix_for([seconds], 's').symbol}s")
    else:
        target = _time_unit(display_unit)
        if target is None:
            raise ValueError(f"pulse display unit must be a time unit: {display_unit!r}")
    return source.scale / target.scale, target.symbol


__all__ = ["pulse_content_bounds", "pulse_time_scale"]
