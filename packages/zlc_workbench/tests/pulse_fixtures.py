"""Ordinary ``zlc.pulse.v1`` JSON fixtures for Workbench tests.

These are deliberately not the Calibration task's template.  They exercise the
normal session/editor pulse path without making Calibration a startup default.
"""

from __future__ import annotations

from pathlib import Path

from zlc_durable import write_readable_json
from zlc_pulse import PulsePeriod, PulseSequence, pulse_target_from_xdc, sequence_to_tree


CAMERA_WINDOWS = 3
PULSE_NAME = "imaging_test"


def _levels(target, *high: str) -> tuple[int, ...]:
    lanes = {
        port.key: port.lanes[0]
        for port in target.ports
        if port.kind == "digital" and len(port.lanes) == 1
    }
    unknown = sorted(set(high) - set(lanes))
    if unknown:
        raise ValueError(f"test pulse target has no digital ports {unknown}")
    positions = {lane: index for index, lane in enumerate(target.raw_lanes)}
    states = [0] * len(target.raw_lanes)
    for key in high:
        states[positions[lanes[key]]] = 1
    return tuple(states)


def ordinary_imaging_sequence(*, name: str = PULSE_NAME) -> PulseSequence:
    """A normal three-window imaging pulse with no Calibration API meaning."""

    target = pulse_target_from_xdc()
    hold = ("trap",)
    image = ("probe", "trap", "emCCD")
    return PulseSequence(
        name=name,
        target=target,
        time_step_ns=20.0,
        periods=(
            PulsePeriod("load", 0.002, "s", _levels(target, "cooling", "trap")),
            PulsePeriod("image_before", 0.020, "s", _levels(target, *image)),
            PulsePeriod("gap_0", 0.0001, "s", _levels(target, *hold)),
            PulsePeriod("image_readout", 0.005, "s", _levels(target, *image)),
            PulsePeriod("gap_1", 0.0151, "s", _levels(target, *hold)),
            PulsePeriod("image_after", 0.020, "s", _levels(target, *image)),
        ),
    )


def write_ordinary_pulse(
    workspace: str | Path,
    *,
    file_stem: str = PULSE_NAME,
    sequence: PulseSequence | None = None,
) -> Path:
    """Write a real v2 ``zlc.pulse.v1`` JSON into an explicit test workspace."""

    pulses = Path(workspace) / "pulses"
    pulses.mkdir(parents=True, exist_ok=True)
    target = pulses / f"{file_stem}.json"
    write_readable_json(target, sequence_to_tree(sequence or ordinary_imaging_sequence()))
    return target


__all__ = [
    "CAMERA_WINDOWS",
    "PULSE_NAME",
    "ordinary_imaging_sequence",
    "write_ordinary_pulse",
]
