"""One pulse, as a tree a file can hold.

A PulseSequence had no persisted form at all, so an editor could change one and
had nowhere to put the change: the Save button was wired to a refusal whose
stated reason -- "a pulse is a Python module, do not overwrite the author's
file" -- was true and answered a different question.  v1 saved a pulse as JSON
beside the module; this is the same fact, owned by the package that owns the
model rather than re-derived by whoever happens to be writing a file.

Trees only.  Reading and writing files belongs to whoever knows where an
experiment keeps things, which is not this package.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .model import (
    AnalogStep,
    OutputDelay,
    PulseFieldRef,
    PulsePeriod,
    PulsePortSpec,
    PulseSequence,
    PulseSlot,
    PulseTarget,
    RepeatRegion,
)


#: What a reader checks before trusting the rest.
PULSE_TREE_FORMAT = "zlc.pulse.v1"


def sequence_to_tree(sequence: PulseSequence) -> dict[str, Any]:
    """Everything needed to rebuild this sequence exactly, as plain data."""

    if not isinstance(sequence, PulseSequence):
        raise TypeError("sequence must be PulseSequence")
    target = sequence.target
    return {
        "format": PULSE_TREE_FORMAT,
        "name": sequence.name,
        "time_step_ns": sequence.time_step_ns,
        "target": {
            "raw_lanes": list(target.raw_lanes),
            "package_pins": dict(target.package_pins),
            "ports": [
                {
                    "key": port.key,
                    "kind": port.kind,
                    "lanes": list(port.lanes),
                    "label": port.label,
                    "bus_index": port.bus_index,
                    "width": port.width,
                    "encoding": port.encoding,
                    "safe_value": port.safe_value,
                    "latch_clock": port.latch_clock,
                }
                for port in target.ports
            ],
        },
        "periods": [
            {
                "period_id": period.period_id,
                "duration": period.duration,
                "unit": period.unit,
                "states": list(period.states),
                "name": period.name,
                "analog_steps": [
                    {"port": step.port, "mode": step.mode, "value": step.value}
                    for step in period.analog_steps
                ],
            }
            for period in sequence.periods
        ],
        "slots": [
            {
                "kind": slot.kind,
                "unit": slot.unit,
                "slot_id": slot.slot_id,
                "field_ref": {
                    "kind": slot.field_ref.kind,
                    "period_id": slot.field_ref.period_id,
                    "port": slot.field_ref.port,
                },
            }
            for slot in sequence.slots
        ],
        "delays": [
            {"port": delay.port, "value": delay.value, "unit": delay.unit}
            for delay in sequence.delays
        ],
        "repeat": (
            None
            if sequence.repeat is None
            else {
                "start_period_id": sequence.repeat.start_period_id,
                "end_period_id": sequence.repeat.end_period_id,
                "count": sequence.repeat.count,
            }
        ),
    }


def sequence_from_tree(tree: Mapping[str, Any]) -> PulseSequence:
    """Rebuild a sequence from :func:`sequence_to_tree`'s output.

    Every value goes back through the model's own constructors, so a tree that
    describes an illegal pulse is refused here rather than becoming one.
    """

    if not isinstance(tree, Mapping):
        raise TypeError("a pulse tree must be a mapping")
    declared = str(tree.get("format", ""))
    if declared != PULSE_TREE_FORMAT:
        raise ValueError(f"not a {PULSE_TREE_FORMAT} pulse: {declared or 'no format'}")

    target_tree = tree["target"]
    target = PulseTarget(
        tuple(str(lane) for lane in target_tree["raw_lanes"]),
        tuple(
            PulsePortSpec(
                key=str(port["key"]),
                kind=str(port["kind"]),
                lanes=tuple(str(lane) for lane in port["lanes"]),
                label=str(port.get("label", "")),
                bus_index=port.get("bus_index"),
                width=port.get("width"),
                encoding=port.get("encoding"),
                safe_value=port.get("safe_value"),
                latch_clock=port.get("latch_clock"),
            )
            for port in target_tree["ports"]
        ),
        package_pins=dict(target_tree.get("package_pins", {})),
    )
    periods = tuple(
        PulsePeriod(
            period_id=str(period["period_id"]),
            duration=period["duration"],
            unit=str(period["unit"]),
            states=tuple(int(state) for state in period["states"]),
            analog_steps=tuple(
                AnalogStep(str(step["port"]), str(step["mode"]), step["value"])
                for step in period.get("analog_steps", ())
            ),
            name=str(period.get("name", "")),
        )
        for period in tree["periods"]
    )
    slots = tuple(
        PulseSlot(
            kind=str(slot["kind"]),
            field_ref=PulseFieldRef(
                kind=str(slot["field_ref"]["kind"]),
                period_id=slot["field_ref"].get("period_id"),
                port=slot["field_ref"].get("port"),
            ),
            unit=str(slot["unit"]),
            slot_id=str(slot["slot_id"]),
        )
        for slot in tree.get("slots", ())
    )
    delays = tuple(
        OutputDelay(str(delay["port"]), delay["value"], str(delay["unit"]))
        for delay in tree.get("delays", ())
    )
    repeat_tree = tree.get("repeat")
    repeat = (
        None
        if repeat_tree is None
        else RepeatRegion(
            str(repeat_tree["start_period_id"]),
            str(repeat_tree["end_period_id"]),
            int(repeat_tree["count"]),
        )
    )
    return PulseSequence(
        name=str(tree.get("name", "sequence")),
        target=target,
        time_step_ns=float(tree["time_step_ns"]),
        periods=periods,
        slots=slots,
        delays=delays,
        repeat=repeat,
    )


__all__ = ["PULSE_TREE_FORMAT", "sequence_from_tree", "sequence_to_tree"]
