"""One pulse, as a tree a file can hold.

A PulseSequence had no persisted form at all, so an editor could change one and
had nowhere to put the change: the Save button was wired to a refusal whose
stated reason -- "a pulse is a Python module, do not overwrite the author's
file" -- was true and answered a different question.  A pulse is saved as JSON
beside the module, with that fact owned by the package that owns the model
rather than re-derived by whoever happens to be writing a file.

Pulse and named Config files share their grammar and file I/O here;
devices and editors do not maintain alternative readers or writers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
from numbers import Real
from typing import Any

from zlc_durable import atomic_write_bytes, readable_json_bytes

from .binding import config_parameter_key

from .model import (
    AnalogStep,
    MAXIMUM_REPEAT_COUNT,
    PERIOD_KIND_PERIOD,
    OutputDelay,
    PulseBinding,
    PulseBracket,
    PulseFieldRef,
    PulsePeriod,
    PulsePortSpec,
    PulseSequence,
    PulseTarget,
)


#: What a reader checks before trusting the rest.
PULSE_TREE_FORMAT = "zlc.pulse"
#: Config keys are cross-pulse names authored in the Config tab.
CONFIG_VALUES_FORMAT = "zlc.pulse.config_values"
CONFIG_VALUES_DIRECTORY = "config_values"
CURRENT_CONFIG_VALUES = "current.json"
PULSE_EDITOR_FIELDS = (
    "visible_ports",
    "scan_source",
    "scan_rows",
    "scan_source_dirty",
    "scan_repeats",
)


def parse_pulse_tree_json(text: str | bytes) -> Mapping[str, Any]:
    """Parse one pulse JSON document without losing malformed input facts."""

    if isinstance(text, bytes):
        text = text.decode("utf-8")
    if not isinstance(text, str):
        raise TypeError("pulse JSON must be text or UTF-8 bytes")

    def object_from_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r} in pulse JSON")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant {value!r} in pulse JSON")

    value = json.loads(
        text,
        object_pairs_hook=object_from_pairs,
        parse_constant=reject_constant,
    )
    if not isinstance(value, Mapping):
        raise TypeError("pulse JSON must contain one object")
    return value


def split_pulse_document_tree(
    tree: Mapping[str, Any],
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    """Split one product pulse document into its sequence and editor sections."""

    if not isinstance(tree, Mapping):
        raise TypeError("pulse document must be an object")
    sequence_tree = dict(tree)
    editor = sequence_tree.pop("editor", {})
    if not isinstance(editor, Mapping):
        raise TypeError("pulse editor state must be an object")
    unknown = tuple(key for key in editor if key not in PULSE_EDITOR_FIELDS)
    if unknown:
        raise ValueError(
            f"unknown pulse editor field(s): {', '.join(map(str, unknown))}"
        )
    visible = editor.get("visible_ports")
    if visible is not None and (
        isinstance(visible, (str, bytes))
        or not isinstance(visible, Sequence)
        or any(not isinstance(key, str) for key in visible)
    ):
        raise TypeError("editor.visible_ports must be null or a list of strings")
    source = editor.get("scan_source", "")
    if not isinstance(source, str):
        raise TypeError("editor.scan_source must be a string")
    rows = editor.get("scan_rows", ())
    if isinstance(rows, (str, bytes, Mapping)) or not isinstance(rows, Sequence):
        raise TypeError("editor.scan_rows must be a table")
    for row in rows:
        if (
            isinstance(row, (str, bytes, Mapping))
            or not isinstance(row, Sequence)
            or any(isinstance(value, bool) or not isinstance(value, Real) for value in row)
        ):
            raise TypeError("each editor.scan_rows row must be a list of numbers")
    dirty = editor.get("scan_source_dirty", False)
    if not isinstance(dirty, bool):
        raise TypeError("editor.scan_source_dirty must be a boolean")
    repeats = editor.get("scan_repeats", 0)
    if isinstance(repeats, bool) or not isinstance(repeats, int):
        raise TypeError("editor.scan_repeats must be an integer")
    if repeats < 0:
        raise ValueError("editor.scan_repeats must be non-negative")
    if repeats > MAXIMUM_REPEAT_COUNT:
        raise ValueError("editor.scan_repeats does not fit the hardware 32-bit count")
    return sequence_tree, editor


def read_pulse_document(
    path: "str | os.PathLike[str]",
) -> tuple[PulseSequence, Mapping[str, Any]]:
    """One pulse from disk, and its editor half.

    THE ONE PLACE A PULSE ARRIVES FROM A FILE.  A config parameter's value is
    the field's own number, so what comes back here is what the pulse was
    saved holding; the sequencer that will play it fills the config
    parameters from the value set it holds, at the moment it compiles.
    """

    source = Path(path).expanduser().resolve()
    sequence_tree, editor = split_pulse_document_tree(
        parse_pulse_tree_json(source.read_text(encoding="utf-8"))
    )
    return sequence_from_tree(sequence_tree), editor


def _object(
    value: Any, expected: tuple[str, ...], name: str, *, optional: tuple[str, ...] = ()
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be text")
    unknown = tuple(key for key in value if key not in expected and key not in optional)
    if unknown:
        raise ValueError(f"unknown {name} field(s): {', '.join(unknown)}")
    missing = tuple(key for key in expected if key not in value)
    if missing:
        raise ValueError(f"missing {name} field(s): {', '.join(missing)}")
    return value


def _array(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON array")
    return value


def sequence_to_tree(sequence: PulseSequence) -> dict[str, Any]:
    """Everything needed to rebuild this sequence exactly, as plain data."""

    if not isinstance(sequence, PulseSequence):
        raise TypeError("sequence must be PulseSequence")
    sequence.require_nonempty_bracket()
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
                "kind": period.kind,
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
        "bindings": [
            {
                "field_ref": {
                    "kind": binding.field_ref.kind,
                    "period_id": binding.field_ref.period_id,
                    "port": binding.field_ref.port,
                },
                "unit": binding.unit,
                "scan": binding.scan,
                "source": binding.source,
                "config_key": binding.config_key,
            }
            for binding in sequence.bindings
        ],
        "delays": [
            {"port": delay.port, "value": delay.value, "unit": delay.unit}
            for delay in sequence.delays
        ],
        "bracket": (
            None
            if sequence.bracket is None
            else {
                "start_period_id": sequence.bracket.start_period_id,
                "end_period_id": sequence.bracket.end_period_id,
                "count": sequence.bracket.count,
            }
        ),
        "run_repeats": sequence.run_repeats,
    }


def sequence_from_tree(tree: Mapping[str, Any]) -> PulseSequence:
    """Rebuild a sequence from :func:`sequence_to_tree`'s output.

    Every value goes back through the model's own constructors, so a tree that
    describes an illegal pulse is refused here rather than becoming one.
    """

    tree = _object(
        tree,
        (
            "format",
            "name",
            "time_step_ns",
            "target",
            "periods",
            "bindings",
            "delays",
            "bracket",
            "run_repeats",
        ),
        "pulse",
    )
    declared = tree["format"]
    if not isinstance(declared, str):
        raise TypeError("pulse format must be text")
    if declared != PULSE_TREE_FORMAT:
        raise ValueError(f"not a {PULSE_TREE_FORMAT} pulse: {declared or 'no format'}")

    target_tree = _object(
        tree["target"],
        ("raw_lanes", "package_pins", "ports"),
        "pulse target",
    )
    package_pins = target_tree["package_pins"]
    if not isinstance(package_pins, Mapping):
        raise TypeError("pulse target package_pins must be an object")
    if any(
        not isinstance(lane, str) or not isinstance(pin, str)
        for lane, pin in package_pins.items()
    ):
        raise TypeError("pulse target package_pins must map text to text")
    target = PulseTarget(
        tuple(_array(target_tree["raw_lanes"], "pulse target raw_lanes")),
        tuple(
            PulsePortSpec(
                key=port["key"],
                kind=port["kind"],
                lanes=tuple(_array(port["lanes"], "pulse port lanes")),
                label=port["label"],
                bus_index=port["bus_index"],
                width=port["width"],
                encoding=port["encoding"],
                safe_value=port["safe_value"],
                latch_clock=port["latch_clock"],
            )
            for port in (
                _object(
                    item,
                    (
                        "key",
                        "kind",
                        "lanes",
                        "label",
                        "bus_index",
                        "width",
                        "encoding",
                        "safe_value",
                        "latch_clock",
                    ),
                    "pulse port",
                )
                for item in _array(target_tree["ports"], "pulse target ports")
            )
        ),
        package_pins=dict(package_pins),
    )
    periods = tuple(
        PulsePeriod(
            period_id=period["period_id"],
            duration=period["duration"],
            unit=period["unit"],
            states=tuple(_array(period["states"], "pulse period states")),
            analog_steps=tuple(
                AnalogStep(step["port"], step["mode"], step["value"])
                for step in (
                    _object(
                        item,
                        ("port", "mode", "value"),
                        "pulse analog step",
                    )
                    for item in _array(
                        period["analog_steps"], "pulse period analog_steps"
                    )
                )
            ),
            name=period["name"],
            kind=period.get("kind", PERIOD_KIND_PERIOD),
        )
        for period in (
            _object(
                item,
                ("period_id", "duration", "unit", "states", "name", "analog_steps"),
                "pulse period",
                optional=("kind",),
            )
            for item in _array(tree["periods"], "pulse periods")
        )
    )
    bindings = []
    for item in _array(tree["bindings"], "pulse bindings"):
        binding = _object(
            item, ("field_ref", "unit", "scan", "source", "config_key"), "pulse binding"
        )
        field = _object(
            binding["field_ref"], ("kind", "period_id", "port"), "pulse field reference"
        )
        bindings.append(PulseBinding(
            field_ref=PulseFieldRef(**field),
            unit=binding["unit"],
            scan=binding["scan"],
            source=binding["source"],
            config_key=binding["config_key"],
        ))
    delays = tuple(
        OutputDelay(delay["port"], delay["value"], delay["unit"])
        for delay in (
            _object(item, ("port", "value", "unit"), "pulse delay")
            for item in _array(tree["delays"], "pulse delays")
        )
    )
    bracket_tree = tree["bracket"]
    bracket = (
        None
        if bracket_tree is None
        else PulseBracket(
            **_object(
                bracket_tree,
                ("start_period_id", "end_period_id", "count"),
                "pulse bracket",
            )
        )
    )
    sequence = PulseSequence(
        name=tree["name"],
        target=target,
        time_step_ns=tree["time_step_ns"],
        periods=periods,
        bindings=tuple(bindings),
        delays=delays,
        bracket=bracket,
        run_repeats=tree["run_repeats"],
    )
    sequence.require_nonempty_bracket()
    return sequence


def _config_value_entry(key: object, number: object, unit: object) -> tuple[float, str]:
    config_parameter_key(key)
    if isinstance(number, bool) or not isinstance(number, Real):
        raise TypeError(f"Config value {key!r} must be a number")
    if not math.isfinite(float(number)):
        raise ValueError(f"Config value {key!r} must be finite")
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError(f"Config value {key!r} must carry a unit")
    return float(number), unit


def config_values_to_tree(
    values: Mapping[str, tuple[int | float, str]],
) -> dict[str, Any]:
    """Serialize the manually authored named Config value table."""

    if not isinstance(values, Mapping):
        raise TypeError("Config values must be a mapping")
    entries = {}
    for key, (number, unit) in values.items():
        number, unit = _config_value_entry(key, number, unit)
        entries[key] = {"value": _plain_number(number), "unit": unit}
    return {"format": CONFIG_VALUES_FORMAT, "values": entries}


def config_values_from_tree(
    tree: Mapping[str, Any],
) -> dict[str, tuple[float, str]]:
    """Read the sole named Config grammar; legacy metadata is not accepted."""

    tree = _object(tree, ("format", "values"), "Config values")
    if tree["format"] != CONFIG_VALUES_FORMAT:
        raise ValueError(f"Config values must declare format {CONFIG_VALUES_FORMAT!r}")
    if not isinstance(tree["values"], Mapping):
        raise TypeError("Config values must be an object")
    entries = {}
    for key, item in tree["values"].items():
        entry = _object(item, ("value", "unit"), f"Config value {key!r}")
        entries[key] = _config_value_entry(key, entry["value"], entry["unit"])
    return entries


def read_config_values(path: str | Path) -> dict[str, tuple[float, str]]:
    """Read a saved Config value table through the same grammar as Save."""

    source = Path(path).expanduser()
    if source.suffix.lower() != ".json":
        raise ValueError(f"config values must be JSON: {source}")
    return config_values_from_tree(parse_pulse_tree_json(source.read_text(encoding="utf-8")))


def write_config_values(
    path: str | Path,
    entries: Mapping[str, tuple[float, str]],
) -> None:
    """Save the Config tab's explicit value table atomically."""

    body = config_values_to_tree(entries)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(target, readable_json_bytes(body))


def _plain_number(value: float) -> int | float:
    return int(value) if float(value).is_integer() else float(value)



__all__ = [
    "CONFIG_VALUES_DIRECTORY",
    "CURRENT_CONFIG_VALUES",
    "CONFIG_VALUES_FORMAT",
    "PULSE_TREE_FORMAT",
    "PULSE_EDITOR_FIELDS",
    "config_values_from_tree",
    "config_values_to_tree",
    "read_config_values",
    "write_config_values",
    "parse_pulse_tree_json",
    "read_pulse_document",
    "sequence_from_tree",
    "sequence_to_tree",
    "split_pulse_document_tree",
]
