"""Offline migration of Pulse bindings, board channels and Config files.

The runtime readers remain strict. Originals are retained beside each migrated
file; numeric Config N becomes the same neutral name ``config_N`` everywhere.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from zlc_durable import atomic_write_bytes, readable_json_bytes, unique_path
from zlc_pulse import PulseTarget, config_values_from_tree, pulse_target_from_xdc, sequence_to_tree
from zlc_pulse.codec import (
    CONFIG_VALUES_FORMAT,
    PULSE_TREE_FORMAT,
    parse_pulse_tree_json,
)

from ..pulse_state import state_from_tree, state_to_tree
from ..session import Workspace


def _number(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"invalid old binding number: {value!r}")
    return value


def migrate_tree(before: Mapping) -> tuple[dict, tuple[str, ...]]:
    """Convert only the known old grammars, then use the production validator."""

    tree = dict(before)
    notes = []
    if tree.get("format") == CONFIG_VALUES_FORMAT:
        unknown = set(tree) - {"format", "name", "source", "values"}
        if unknown:
            raise ValueError(f"unknown Config fields: {sorted(unknown)}")
        values = {}
        for key, entry in tree["values"].items():
            if not isinstance(entry, Mapping) or set(entry) - {"value", "unit", "field"}:
                raise ValueError(f"unknown Config entry fields: {key!r}")
            converted = key
            if key.isascii() and key.isdecimal():
                number = _number(int(key))
                if str(number) != key:
                    raise ValueError(f"noncanonical old Config number: {key!r}")
                converted = f"config_{number}"
                notes.append(f"Config {key} -> {converted}")
            if converted in values:
                raise ValueError(f"Config name collision: {converted}")
            values[converted] = {"value": entry["value"], "unit": entry["unit"]}
        tree = {"format": CONFIG_VALUES_FORMAT, "values": values}
        config_values_from_tree(tree)
    elif tree.get("format") == PULSE_TREE_FORMAT:
        old_names = ("slots", "api_parameters", "config_parameters")
        if any(name in tree for name in old_names):
            if "bindings" in tree:
                raise ValueError("Pulse mixes old and current bindings")
            bindings = []
            identities = set()
            for category in old_names:
                items = tree.pop(category, [])
                if not isinstance(items, list):
                    raise ValueError(f"{category} must be a list")
                numbers = [_number(item["number"]) for item in items if item.get("number") is not None]
                if len(numbers) != len(set(numbers)):
                    raise ValueError(f"duplicate numbers in {category}")
                assigned = set(numbers)
                for item in items:
                    scan = category == "slots"
                    identity_key = "slot_id" if scan else "parameter_id"
                    required = {identity_key, "unit", "field_ref"} | ({"kind"} if scan else set())
                    if not isinstance(item, Mapping) or set(item) - required - {"number"} or required - set(item):
                        raise ValueError(f"unsupported old {category} entry")
                    if scan and item["kind"] != item["field_ref"].get("kind"):
                        raise ValueError("old Scan kind and physical field differ")
                    identity = item[identity_key]
                    if not isinstance(identity, str) or not identity or identity in identities:
                        raise ValueError(f"invalid or duplicate binding identity: {identity!r}")
                    identities.add(identity)
                    number = item.get("number")
                    if number is None:
                        # Exact old PulseSequence rule: smallest free number,
                        # preserving explicit assignments and document order.
                        number = 1
                        while number in assigned:
                            number += 1
                        assigned.add(number)
                        notes.append(f"{identity}: old implicit number {number}")
                    source = {"slots": "default", "api_parameters": "api", "config_parameters": "config"}[category]
                    key = f"config_{number}" if source == "config" else ""
                    if key:
                        notes.append(f"{identity}: Config {number} -> {key}")
                    bindings.append({
                        "field_ref": item["field_ref"],
                        "unit": item["unit"],
                        "scan": scan,
                        "source": source,
                        "config_key": key,
                    })
            tree["bindings"] = bindings
            notes.append("physical Scan/API fields; named Config bindings")
        if "repeat" in tree and "bracket" not in tree:
            tree["bracket"] = tree.pop("repeat")
        tree.setdefault("run_repeats", 0)
        if "config_source" in tree:
            tree.pop("config_source")
            notes.append("removed implicit Config path; explicitly Load Config before Fire")
        state = state_from_tree(tree)
        sequence = state.sequence
        target = pulse_target_from_xdc()
        if len(sequence.target.raw_lanes) == 63 and len(target.raw_lanes) == 69:
            old = sequence.target
            if old.raw_lanes != tuple(f"ch{index:02d}" for index in range(63)):
                raise ValueError("not the known 63-lane board; raw lanes cannot be inferred")
            if old.package_pins:
                by_pin = {pin: lane for lane, pin in target.package_pins.items()}
                try:
                    lanes = {lane: by_pin[pin] for lane, pin in old.package_pins.items()}
                except KeyError as error:
                    raise ValueError(f"old board pin is absent from the current manifest: {error}") from error
            else:
                # Known pre-expansion board only: the complete port geometry
                # below must match, not merely the number of state entries.
                lanes = {lane: target.raw_lanes[index if index < 19 else index + 6]
                         for index, lane in enumerate(old.raw_lanes)}
            port_names = {port.key: "shutter_420" if port.key == "cooling_pgc" else port.key
                          for port in old.ports}
            for port in old.ports:
                candidate = target.by_key.get(port_names[port.key])
                if (candidate is None or candidate.kind != port.kind
                    or candidate.lanes != tuple(lanes[lane] for lane in port.lanes)
                    or (candidate.bus_index, candidate.width, candidate.encoding, candidate.safe_value,
                        candidate.latch_clock) != (port.bus_index, port.width, port.encoding, port.safe_value,
                                                  port.latch_clock)):
                    raise ValueError(f"old port {port.key!r} does not match the known board wiring")
            if len(old.ports) != len(target.ports) - 6:
                raise ValueError("old board does not declare every pre-expansion port")
            labels = {port_names[port.key]: port.label for port in old.ports
                      if port.label != port.key}
            if labels:
                target = PulseTarget(
                    target.raw_lanes,
                    tuple(replace(port, label=labels.get(port.key, port.label)) for port in target.ports),
                    package_pins=target.package_pins,
                )
            indices = {lane: index for index, lane in enumerate(target.raw_lanes)}
            periods = []
            for period in sequence.periods:
                states = [0] * len(target.raw_lanes)
                for lane, value in zip(old.raw_lanes, period.states, strict=True):
                    states[indices[lanes[lane]]] = value
                periods.append(replace(period, states=tuple(states)))
            sequence = replace(
                sequence, target=target, periods=tuple(periods),
                bindings=tuple(replace(binding, field_ref=replace(
                    binding.field_ref,
                    port=None if binding.field_ref.port is None else port_names[binding.field_ref.port],
                )) for binding in sequence.bindings),
                delays=tuple(replace(delay, port=port_names[delay.port]) for delay in sequence.delays),
            )
            # Scan columns retain their binding order and numeric table. Its
            # Python source produces positional columns, not port-key lookups;
            # preserve authored code instead of replacing arbitrary strings.
            if "editor" in tree:
                tree = state_to_tree(replace(
                    state, sequence=sequence,
                    visible_ports=(None if state.visible_ports is None else frozenset(port_names[key] for key in state.visible_ports)),
                ))
            else:
                tree = sequence_to_tree(sequence)
            state_from_tree(tree)
            notes.append("63 -> 69 physical lanes; F13 cooling_pgc -> shutter_420; six new TTL outputs low")
        elif (sequence.target.abi_fingerprint != target.abi_fingerprint
              or (sequence.target.package_pins and sequence.target.package_pins != target.package_pins)):
            notes.append("board target left unchanged: not this board's known predecessor")
    else:
        raise ValueError("not a Pulse or Config document")
    return tree, tuple(notes)


def migrate_file(path: Path, *, dry_run: bool = False) -> str:
    raw = path.read_bytes()
    before = parse_pulse_tree_json(raw)
    if before.get("format") not in (PULSE_TREE_FORMAT, CONFIG_VALUES_FORMAT):
        return "SKIP (not Pulse/Config)"
    after, notes = migrate_tree(before)
    if before == after:
        return f"SKIP: {'; '.join(notes)}" if notes else "CURRENT"
    detail = "; ".join(notes) or "removed obsolete Config metadata"
    if dry_run:
        return f"WOULD MIGRATE: {detail}"
    payload = readable_json_bytes(after)
    # Both the candidate and the original bytes are complete before replacement.
    backup = unique_path(
        path.parent,
        path.name,
        ".pre-binding-migration",
        writer=lambda target: target.write_bytes(raw),
    )
    atomic_write_bytes(path, payload)
    return f"MIGRATED: {detail}; original: {backup.name}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="Pulse/Config JSON files or folders (recursive)")
    parser.add_argument("--dry-run", action="store_true", help="validate and report without writing")
    args = parser.parse_args(argv)
    if args.paths:
        roots = [path.expanduser().resolve() for path in args.paths]
    else:
        workspace = Workspace.discover()
        roots = [workspace.pulses, workspace.config_values]
    paths = set()
    failures = 0
    for root in roots:
        print(f"Selected: {root}")
        if not root.exists():
            print("  No such path")
            failures += bool(args.paths)
        elif root.is_file():
            paths.add(root)
        else:
            # A selected directory never authorizes files reached through a
            # symlink outside it or any sibling directory.
            paths.update(path for path in root.rglob("*.json") if path.resolve().is_relative_to(root))
    for path in sorted(paths):
        try:
            print(f"{path}: {migrate_file(path, dry_run=args.dry_run)}")
        except Exception as error:
            failures += 1
            print(f"{path}: FAILED: {error}")
    print("Config numbers become config_N. Load the migrated Config explicitly; runtime never auto-loads current.json.")
    print(f"Finished: {len(paths)} file(s), {failures} failure(s). Originals are never deleted.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
