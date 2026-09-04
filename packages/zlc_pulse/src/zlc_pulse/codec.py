"""One pulse, as a tree a file can hold.

A PulseSequence had no persisted form at all, so an editor could change one and
had nowhere to put the change: the Save button was wired to a refusal whose
stated reason -- "a pulse is a Python module, do not overwrite the author's
file" -- was true and answered a different question.  A pulse is saved as JSON
beside the module, with that fact owned by the package that owns the model
rather than re-derived by whoever happens to be writing a file.

Trees only, with one exception that earns itself: :func:`read_pulse_document`.
A pulse names the file it refreshes its config parameters from, and it names
it RELATIVE TO ITSELF -- so the only thing that knows where that file is, is
whoever is holding the pulse's own path.  Pushing the read outward would hand
every consumer the same three lines to remember, and the one that forgot would
play last month's bias without saying so.  Everything else here is still trees.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
from numbers import Real
from typing import Any

from .binding import apply_config_values
from .model import (
    AnalogStep,
    MAXIMUM_REPEAT_COUNT,
    OutputDelay,
    PulseApiParameter,
    PulseConfigParameter,
    PulseBracket,
    PulseFieldRef,
    PulsePeriod,
    PulsePortSpec,
    PulseSequence,
    PulseSlot,
    PulseTarget,
)


#: What a reader checks before trusting the rest.
PULSE_TREE_FORMAT = "zlc.pulse"
#: ...and the same for one saved set of API parameter values.
API_VALUES_FORMAT = "zlc.pulse.api_values"
#: ...and for a set of CONFIG parameter values, which is a different kind of
#: file even though it holds the same shape of numbers: an API set is chosen
#: for one run, a config set is what a pulse refreshes itself from.  Separate
#: roots so neither can be handed to the other by mistake.
CONFIG_VALUES_FORMAT = "zlc.pulse.config_values"
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
    """One pulse from disk -- config values refreshed -- and its editor half.

    THE ONE PLACE A PULSE ARRIVES FROM A FILE.  A config parameter's value is
    the field's own number, so nothing downstream has to resolve anything: by
    the time the pulse leaves here it already holds today's numbers, and a
    runner that never heard of config parameters plays them correctly without
    a line of its own.  That is the whole point of putting the read here
    rather than at each of the places that fire a pulse.

    A pulse that declares no config parameter never looks for a file.  One
    that declares them and names no source keeps its authored numbers -- it
    refreshes from nowhere, which is a legal thing to be.  Otherwise the file
    is read, and anything wrong with it -- missing, unreadable, not a config
    set, or silent about a parameter this pulse declares -- is raised here,
    before the pulse reaches anything that could play it.
    """

    source = Path(path).expanduser().resolve()
    sequence_tree, editor = split_pulse_document_tree(
        parse_pulse_tree_json(source.read_text(encoding="utf-8"))
    )
    return refreshed_config_values(sequence_from_tree(sequence_tree), source), editor


def refreshed_config_values(
    sequence: PulseSequence, pulse_path: "str | os.PathLike[str]"
) -> PulseSequence:
    """This pulse with its config parameters set to what its config file says.

    Public because two callers need the same rule and must not each have
    their own: every read of a pulse from disk, and the Pulse Editor's
    Refresh, which re-pulls without reopening the document the operator
    may have edited.
    """

    pulse_path = Path(pulse_path)

    if not sequence.config_parameters or not sequence.config_source:
        return sequence
    source = (pulse_path.parent / sequence.config_source).expanduser()
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(
            f"{sequence.name!r} refreshes its config from {sequence.config_source!r}, "
            f"which cannot be read: {error}"
        ) from None
    try:
        _name, _origin, entries = config_values_from_tree(json.loads(text))
    except Exception as error:
        raise ValueError(
            f"{sequence.name!r} refreshes its config from {sequence.config_source!r}, "
            f"which is not a config value set: {error}"
        ) from None
    refreshed, _applied, _unknown = apply_config_values(sequence, entries)
    absent = tuple(
        parameter.parameter_id
        for parameter in sequence.config_parameters
        if parameter.parameter_id not in entries
    )
    if absent:
        # Silence here would run a stale number while the pulse says it is
        # fresh, which is the one outcome worse than refusing to run.
        raise ValueError(
            f"{sequence.name!r} declares config parameter(s) {absent} that "
            f"{sequence.config_source!r} says nothing about"
        )
    return refreshed


def _object(value: Any, expected: tuple[str, ...], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be text")
    unknown = tuple(key for key in value if key not in expected)
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
        "api_parameters": [
            _named_binding_tree(parameter) for parameter in sequence.api_parameters
        ],
        "config_parameters": [
            _named_binding_tree(parameter)
            for parameter in sequence.config_parameters
        ],
        "config_source": sequence.config_source,
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


def _named_binding_tree(binding: Any) -> dict[str, Any]:
    """One named field binding -- API or config -- as a tree."""

    return {
        "parameter_id": binding.parameter_id,
        "unit": binding.unit,
        "field_ref": {
            "kind": binding.field_ref.kind,
            "period_id": binding.field_ref.period_id,
            "port": binding.field_ref.port,
        },
    }


def _named_bindings(items: Any, factory: Any, label: str) -> tuple[Any, ...]:
    """Rebuild one list of named field bindings through the model's own types."""

    rebuilt = []
    for item in _array(items, f"pulse {label}s"):
        binding = _object(item, ("parameter_id", "unit", "field_ref"), f"pulse {label}")
        field = _object(
            binding["field_ref"],
            ("kind", "period_id", "port"),
            "pulse field reference",
        )
        rebuilt.append(
            factory(
                parameter_id=binding["parameter_id"],
                field_ref=PulseFieldRef(
                    kind=field["kind"],
                    period_id=field["period_id"],
                    port=field["port"],
                ),
                unit=binding["unit"],
            )
        )
    return tuple(rebuilt)


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
            "slots",
            "api_parameters",
            "config_parameters",
            "config_source",
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
        )
        for period in (
            _object(
                item,
                ("period_id", "duration", "unit", "states", "name", "analog_steps"),
                "pulse period",
            )
            for item in _array(tree["periods"], "pulse periods")
        )
    )
    slots = tuple(
        PulseSlot(
            kind=slot["kind"],
            field_ref=PulseFieldRef(
                kind=field["kind"],
                period_id=field["period_id"],
                port=field["port"],
            ),
            unit=slot["unit"],
            slot_id=slot["slot_id"],
        )
        for slot, field in (
            (
                slot,
                _object(
                    slot["field_ref"],
                    ("kind", "period_id", "port"),
                    "pulse field reference",
                ),
            )
            for slot in (
                _object(
                    item,
                    ("kind", "unit", "slot_id", "field_ref"),
                    "pulse slot",
                )
                for item in _array(tree["slots"], "pulse slots")
            )
        )
    )
    api_parameters = _named_bindings(
        tree["api_parameters"], PulseApiParameter, "API parameter"
    )
    config_parameters = _named_bindings(
        tree["config_parameters"], PulseConfigParameter, "config parameter"
    )
    config_source = tree["config_source"]
    if not isinstance(config_source, str):
        raise TypeError("pulse config_source must be text")
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
    return PulseSequence(
        name=tree["name"],
        target=target,
        time_step_ns=tree["time_step_ns"],
        periods=periods,
        slots=slots,
        api_parameters=api_parameters,
        config_parameters=config_parameters,
        config_source=config_source,
        delays=delays,
        bracket=bracket,
        run_repeats=tree["run_repeats"],
    )


def api_values_to_tree(
    values: Mapping[str, tuple[int | float, str]],
    *,
    name: str = "",
    source: str = "hand",
) -> dict[str, Any]:
    """One named set of API parameter values, as a tree a file can hold.

    Written from a pulse, so the ids and units are the pulse's own and an
    operator editing the numbers by hand cannot invent a name that nothing
    declares.  ``source`` is free text: whoever produced these numbers stamps
    itself there, and a reader does not care who that was.
    """

    return {
        **_named_values_tree(values, "API value"),
        "format": API_VALUES_FORMAT,
        "name": str(name),
        "source": str(source),
    }


def _named_values_tree(
    values: Mapping[str, tuple[int | float, str]], label: str
) -> dict[str, Any]:
    """The ``values`` body both value-set grammars carry."""

    if not isinstance(values, Mapping):
        raise TypeError(f"{label}s must be a mapping")
    entries: dict[str, Any] = {}
    for parameter_id, entry in values.items():
        number, unit = entry
        if not isinstance(parameter_id, str) or not parameter_id:
            raise ValueError(f"{label} ids must be non-empty text")
        if not isinstance(number, Real) or isinstance(number, bool):
            raise TypeError(f"{label} {parameter_id!r} must be a number")
        if not isinstance(unit, str) or not unit:
            raise ValueError(f"{label} {parameter_id!r} must carry a unit")
        entries[parameter_id] = {"value": _plain_number(float(number)), "unit": unit}
    return {"values": entries}


def api_values_from_tree(
    tree: Mapping[str, Any],
) -> tuple[str, str, dict[str, tuple[float, str]]]:
    """``(name, source, {parameter_id: (value, unit)})`` from a saved set.

    The grammar is closed the way a pulse's is: an alternate root, a missing
    field or an unknown one is refused here rather than becoming a silently
    half-applied set of values.
    """

    return _named_values_from_tree(tree, API_VALUES_FORMAT, "API value")


def _named_values_from_tree(
    tree: Mapping[str, Any], declared_format: str, label: str
) -> tuple[str, str, dict[str, tuple[float, str]]]:
    """One saved value set, read under a closed grammar."""

    tree = _object(tree, ("format", "name", "source", "values"), f"{label}s")
    if tree["format"] != declared_format:
        raise ValueError(f"{label}s must declare format {declared_format!r}")
    for field in ("name", "source"):
        if not isinstance(tree[field], str):
            raise TypeError(f"{label}s {field} must be text")
    body = tree["values"]
    if not isinstance(body, Mapping):
        raise TypeError(f"{label}s body must be an object")
    entries: dict[str, tuple[float, str]] = {}
    for parameter_id, entry in body.items():
        if not isinstance(parameter_id, str) or not parameter_id:
            raise ValueError(f"{label} ids must be non-empty text")
        entry = _object(entry, ("value", "unit"), f"{label} {parameter_id!r}")
        number = entry["value"]
        if not isinstance(number, Real) or isinstance(number, bool):
            raise TypeError(f"{label} {parameter_id!r} must be a number")
        unit = entry["unit"]
        if not isinstance(unit, str) or not unit:
            raise ValueError(f"{label} {parameter_id!r} must carry a unit")
        entries[parameter_id] = (float(number), unit)
    return tree["name"], tree["source"], entries


def config_values_to_tree(
    values: Mapping[str, tuple[int | float, str]],
    *,
    name: str = "",
    source: str = "hand",
) -> dict[str, Any]:
    """One named set of CONFIG parameter values, as a tree a file can hold.

    What a calibration writes and a pulse refreshes itself from.  Same shape
    as an API set and deliberately not the same root: handing one to the other
    is a mistake a reader should catch, not carry out.
    """

    return {
        **_named_values_tree(values, "config value"),
        "format": CONFIG_VALUES_FORMAT,
        "name": str(name),
        "source": str(source),
    }


def config_values_from_tree(
    tree: Mapping[str, Any],
) -> tuple[str, str, dict[str, tuple[float, str]]]:
    """``(name, source, {parameter_id: (value, unit)})`` from a saved config set."""

    return _named_values_from_tree(tree, CONFIG_VALUES_FORMAT, "config value")


def _plain_number(value: float) -> int | float:
    return int(value) if float(value).is_integer() else float(value)



__all__ = [
    "API_VALUES_FORMAT",
    "PULSE_TREE_FORMAT",
    "PULSE_EDITOR_FIELDS",
    "api_values_from_tree",
    "api_values_to_tree",
    "parse_pulse_tree_json",
    "read_pulse_document",
    "refreshed_config_values",
    "sequence_from_tree",
    "sequence_to_tree",
    "split_pulse_document_tree",
]
