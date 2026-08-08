"""The readout calibration bracket: long, short, long, after one MOT load.

The physics this encodes, and why the shape matters: the two LONG windows either
side of the short one form a consensus label pair.  An atom counts as present
for this shot only if BOTH long frames see it, which is what proves it was there
for the whole bracket rather than arriving or leaving midway.  The short frame
in between is the one being characterised.  Two separate loads, or a bracket
with only two windows, destroy that consensus and poison every label.

One sequencer fire is one calibration shot and produces three camera frames.

Board mapping, geometry and clock all come from the board itself -- the XDC the
bitstream was built from, and the streamer config beside it.  Nothing here
hand-writes a lane number: channels are named the way the board names them.
"""

from __future__ import annotations

import math

from zlc_pulse import (
    PulsePeriod,
    PulseSequence,
    compile_sequence,
    load_streamer_config,
    pulse_target_from_xdc,
)


LONG_SECONDS = 0.020
SHORT_SECONDS = 0.005

#: The camera window channel, named as the board names it.
CAMERA_CHANNEL = "emCCD"

_HOLD = ("trap",)
_IMAGE = ("probe", "trap", CAMERA_CHANNEL)


def _levels(target, *high: str) -> tuple[int, ...]:
    """One lane vector with ``high`` asserted, addressed by board signal name."""

    single_lane = {
        port.key: port.lanes[0] for port in target.ports if len(port.lanes) == 1
    }
    unknown = sorted(set(high) - set(single_lane))
    if unknown:
        raise ValueError(f"the board declares no digital channel named {unknown}")

    position = {lane: index for index, lane in enumerate(target.raw_lanes)}
    levels = [0] * len(target.raw_lanes)
    for name in high:
        levels[position[single_lane[name]]] = 1
    return tuple(levels)


def build(
    *,
    reference_exposure_seconds: float = LONG_SECONDS,
    readout_exposure_seconds: float = SHORT_SECONDS,
) -> tuple[object, dict[str, object]]:
    """Compile the bracket for the deployed board and describe its frames."""

    reference_exposure = float(reference_exposure_seconds)
    readout_exposure = float(readout_exposure_seconds)
    if not math.isfinite(reference_exposure) or reference_exposure <= 0:
        raise ValueError("reference_exposure_seconds must be positive and finite")
    if not math.isfinite(readout_exposure) or readout_exposure <= 0:
        raise ValueError("readout_exposure_seconds must be positive and finite")
    if readout_exposure > reference_exposure:
        raise ValueError("readout exposure cannot exceed reference exposure")

    config = load_streamer_config()
    if config["source"] is None:
        raise RuntimeError(
            "no streamer config was found, so the deployed board geometry is "
            "unknown; refusing to compile against built-in defaults"
        )

    target = pulse_target_from_xdc()
    sequence = PulseSequence(
        name="calibration",
        target=target,
        time_step_ns=20.0,
        periods=(
            PulsePeriod("load", 0.002, "s", _levels(target, "cooling", "trap")),
            PulsePeriod("long_before", reference_exposure, "s", _levels(target, *_IMAGE)),
            PulsePeriod("gap_0", 0.0001, "s", _levels(target, *_HOLD)),
            PulsePeriod("short", readout_exposure, "s", _levels(target, *_IMAGE)),
            PulsePeriod("gap_1", 0.0151, "s", _levels(target, *_HOLD)),
            PulsePeriod("long_after", reference_exposure, "s", _levels(target, *_IMAGE)),
        ),
    )
    program = compile_sequence(sequence, config["params"], config["clock_hz"])
    # Counted off the program, not written down beside it.  A hand-written 3
    # is a second answer to a question the compiled edges already settle, and
    # the two can only ever agree by luck after the next edit.
    return program, {
        "camera_trigger_channel": CAMERA_CHANNEL,
        "camera_windows": program.camera_window_count(CAMERA_CHANNEL),
        "frame_exposures": program.camera_window_exposures(CAMERA_CHANNEL),
        "frame_semantics": (
            "reference_long_before",
            "short_readout",
            "reference_long_after",
        ),
        "reference_frame_indices": (0, 2),
        "short_frame_index": 1,
        "exposure_seconds": readout_exposure,
        "sequence": sequence,
    }


__all__ = ["build"]
