"""The explicit offline tool preserves Pulse values and Config relationships."""

from copy import deepcopy
from dataclasses import replace
import json

import pytest

from pulse_fixtures import ordinary_imaging_sequence
from zlc_pulse import apply_config_values, pulse_field_value, pulse_target_from_xdc, read_config_values, sequence_from_tree, sequence_to_tree
from zlc_workbench.pulse_state import read_pulse
from zlc_workbench.tools.migrate_pulses import main, migrate_file, migrate_tree


def test_offline_migration_preserves_numeric_config_and_scan_order(tmp_path):
    tree = sequence_to_tree(ordinary_imaging_sequence())
    tree.pop("bindings")
    periods = tree["periods"]
    fields = [dict(kind="duration", period_id=p["period_id"], port=None) for p in periods[:3]]
    tree["slots"] = [{"kind": "duration", "unit": periods[0]["unit"], "slot_id": "scan0", "field_ref": fields[0], "number": 3}]
    tree["api_parameters"] = [{"unit": periods[1]["unit"], "parameter_id": "exposure", "field_ref": fields[1], "number": 5}]
    tree["config_parameters"] = [{"unit": periods[2]["unit"], "parameter_id": "stored", "field_ref": fields[2]}]
    tree["editor"] = {"scan_rows": [[periods[0]["duration"]]], "scan_repeats": 2}
    pulse = tmp_path / "pulse.json"
    original = json.dumps(tree).encode()
    pulse.write_bytes(original)
    config = tmp_path / "current.json"
    config_original = json.dumps({
        "format": "zlc.pulse.config_values", "name": "current", "source": "hand",
        "values": {"1": {"value": 7, "unit": "ms", "field": "old label"}},
    }).encode()
    config.write_bytes(config_original)
    unrelated = tmp_path / "apparatus.json"
    unrelated.write_text('{"format":"apparatus","devices":[]}', encoding="utf-8")
    assert main([str(tmp_path), "--dry-run"]) == 0
    assert pulse.read_bytes() == original and config.read_bytes() == config_original
    assert not list(tmp_path.glob("*.pre-binding-migration"))
    assert main([str(tmp_path)]) == 0
    result = read_pulse(pulse)
    assert sequence_to_tree(result.sequence)["periods"] == periods
    assert result.sequence.scan_bindings[0].field_ref.period_id == periods[0]["period_id"]
    assert result.sequence.api_bindings[0].field_ref.period_id == periods[1]["period_id"]
    assert result.sequence.config_bindings[0].config_key == "config_1"
    assert result.scan_rows == ((periods[0]["duration"],),)
    assert read_config_values(config) == {"config_1": (7.0, "ms")}
    applied, _, _ = apply_config_values(result.sequence, read_config_values(config))
    assert pulse_field_value(applied, result.sequence.config_bindings[0].field_ref, "ms") == 7
    assert (tmp_path / "pulse.json.pre-binding-migration").read_bytes() == original
    assert (tmp_path / "current.json.pre-binding-migration").read_bytes() == config_original
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert main([str(tmp_path)]) == 0
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before

    # Preserve the old smallest-unused numbering rule, not parameter order+1.
    legacy = deepcopy(tree)
    legacy["config_parameters"].append({
        "parameter_id": "explicit", "number": 1,
        "unit": "ns", "field_ref": fields[1],
    })
    legacy["api_parameters"] = []
    converted, notes = migrate_tree(legacy)
    assert [item["config_key"] for item in converted["bindings"] if item["source"] == "config"] == ["config_2", "config_1"]
    assert any("implicit number 2" in note for note in notes)

    # The board expansion preserves old physical pins, including DAC bit
    # order, while only F13's logical label changes. No user file is touched.
    target = pulse_target_from_xdc()
    assert len(target.raw_lanes) == 69
    old_board = sequence_to_tree(ordinary_imaging_sequence())
    target_tree = old_board["target"]
    old_to_new = {f"ch{i:02d}": f"ch{i if i < 19 else i + 6:02d}" for i in range(63)}
    new_to_old = {new: old for old, new in old_to_new.items()}
    target_tree["raw_lanes"] = list(old_to_new)
    target_tree["package_pins"] = {old: target.package_pins[new] for old, new in old_to_new.items()}
    ports = []
    for port in target_tree["ports"]:
        if port["lanes"][0] not in new_to_old:
            continue
        port["lanes"] = [new_to_old[lane] for lane in port["lanes"]]
        if port["key"] == "shutter_420":
            port["key"] = port["label"] = "cooling_pgc"
        elif port["key"] == "trig":
            port["label"] = "External sequence trigger"
        ports.append(port)
    target_tree["ports"] = ports
    for period in old_board["periods"]:
        period["states"] = period["states"][:19] + period["states"][25:]
    old_board["periods"][0]["states"][1] = 1
    old_board["periods"][0]["states"][6] = 1
    old_board["periods"][0]["states"][18] = 1
    old_board["periods"][0]["analog_steps"] = [{"port": "da_bias_y", "mode": "edge", "value": 17}]
    old_board["delays"] = [{"port": "cooling_pgc", "value": 20, "unit": "ns"}]
    old_board["bindings"] = [
        {"field_ref": {"kind": "delay", "period_id": None, "port": "cooling_pgc"},
         "unit": "ns", "scan": False, "source": "config", "config_key": "cooling_pgc"},
        {"field_ref": {"kind": "dac", "period_id": "load", "port": "da_bias_y"},
         "unit": "value", "scan": True, "source": "api", "config_key": ""},
    ]
    old_board["editor"] = {"visible_ports": ["cooling_pgc", "trig", "da_bias_y"],
                           "scan_source": "# cooling_pgc is an authored comment\nscan_table = [[17], [18]]\n",
                           "scan_rows": [[17], [18]], "scan_source_dirty": False, "scan_repeats": 3}
    for with_pins in (True, False):
        source_tree = deepcopy(old_board)
        if not with_pins:
            source_tree["target"]["package_pins"] = {}
        source_path = tmp_path / f"old-board-{with_pins}.json"
        source_bytes = json.dumps(source_tree).encode()
        source_path.write_bytes(source_bytes)
        assert main([str(source_path), "--dry-run"]) == 0
        assert source_path.read_bytes() == source_bytes
        assert main([str(source_path)]) == 0
        migrated = read_pulse(source_path)
        assert migrated.sequence.target.raw_lanes == target.raw_lanes
        assert migrated.sequence.target.package_pins == target.package_pins
        assert migrated.sequence.target.ports == tuple(
            replace(port, label="External sequence trigger") if port.key == "trig" else port
            for port in target.ports
        )
        for previous, current in zip(old_board["periods"], migrated.sequence.periods, strict=True):
            for old_index, new_lane in enumerate(old_to_new.values()):
                assert current.states[target.raw_lanes.index(new_lane)] == previous["states"][old_index]
            assert current.states[19:25] == (0,) * 6
        assert migrated.sequence.periods[0].analog_steps[0].value == 17
        assert migrated.sequence.delays[0].port == "shutter_420"
        assert migrated.sequence.config_bindings[0].field_ref.port == "shutter_420"
        assert migrated.sequence.config_bindings[0].config_key == "cooling_pgc"
        assert migrated.visible_ports == frozenset(("shutter_420", "trig", "da_bias_y"))
        assert migrated.scan_rows == ((17,), (18,))
        assert migrated.scan_source == old_board["editor"]["scan_source"]
        assert migrated.scan_repeats == 3 and not migrated.scan_source_dirty
        backup = source_path.with_name(source_path.name + ".pre-binding-migration")
        assert backup.read_bytes() == source_bytes
        after = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
        assert main([str(source_path)]) == 0
        assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == after
    invalid_pin = deepcopy(old_board)
    invalid_pin["target"]["package_pins"]["ch00"] = "UNKNOWN_PIN"
    with pytest.raises(ValueError, match="pin is absent"):
        migrate_tree(invalid_pin)
    pure_pulse = deepcopy(old_board)
    pure_pulse.pop("editor")
    upgraded, _notes = migrate_tree(pure_pulse)
    assert "editor" not in upgraded
    assert len(sequence_from_tree(upgraded).target.raw_lanes) == 69
    # No false "CURRENT" declaration or edit for an unrelated logical board.
    other = sequence_to_tree(ordinary_imaging_sequence())
    other["target"] = {"raw_lanes": ["output"], "package_pins": {}, "ports": [{
        "key": "output", "kind": "digital", "lanes": ["output"], "label": "Output",
        "bus_index": None, "width": 1, "encoding": "binary", "safe_value": 0, "latch_clock": None,
    }]}
    for period in other["periods"]:
        period["states"] = [0]
        period["analog_steps"] = []
    path = tmp_path / "other-board.json"
    raw = json.dumps(other).encode()
    path.write_bytes(raw)
    assert migrate_file(path).startswith("SKIP: board target left unchanged")
    assert path.read_bytes() == raw
    assert not path.with_name(path.name + ".pre-binding-migration").exists()


def test_migration_refuses_ambiguous_documents_without_touching_original(tmp_path):
    config = {"format": "zlc.pulse.config_values", "name": "old", "source": "hand",
              "values": {"1": {"value": 2, "unit": "ms"}, "config_1": {"value": 3, "unit": "ms"}}}
    path = tmp_path / "conflict.json"
    raw = json.dumps(config).encode()
    path.write_bytes(raw)
    assert main([str(path)]) == 1
    assert path.read_bytes() == raw
    assert list(tmp_path.iterdir()) == [path]
    with pytest.raises(ValueError, match="collision"):
        migrate_tree(config)
