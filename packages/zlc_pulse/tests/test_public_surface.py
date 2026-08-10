from __future__ import annotations

import builtins
import importlib
import inspect
from pathlib import Path
import re

import zlc_pulse
from zlc_pulse import remote, wire
from zlc_pulse.remote import REMOTE_METHODS, RemotePulseStreamer


#: Raised deliberately, each time for a capability that was MISSING:
#: MINIMUM_REPEAT_COUNT, because the editor invented its own answer to "what
#: is the smallest count that IS a repeat" and got it wrong; and the three
#: codec names, because a PulseSequence had no persisted form at all.
#:
#: 26 -> 34 for the eight the editor was reaching into submodules for, each
#: for a capability the facade did not otherwise offer: TIME_UNIT_CHOICES /
#: TIME_UNIT_TO_NS (which units a duration may use, and what they are worth in
#: ticks -- the editor cannot offer a unit box without them); DEFAULT_HOST /
#: DEFAULT_PORT (where a board answers, written in five places across four
#: packages before it had one); the three scan names (what columns a bound
#: pulse has, a starting table, and whether a table is legal -- the Scan page
#: is those three questions); and MemoryRegisterTransport, which is how a
#: board is driven with no board present.
#:
#: Every one is declared here and in the runtime contract.  A sibling package
#: reaching into a submodule for something this list does not name is the same
#: surface with the declaration skipped, which is worse than a wide facade --
#: nobody can see how wide it really is.
#: 34 -> 36 for the scan crossing.  A table is written in the units the editor
#: shows and held in the units the wire uses, and until now the difference was
#: the author's problem: the template offered offset-binary codes and device
#: ticks for fields whose own boxes are signed codes and microseconds.  Naming
#: both directions makes the conversion a place instead of an instruction.
#: 36 -> 38 for the two finite authoring facts a sibling editor consumes:
#: analog step choices and the legal binding transition for a physical field.
#: Keeping them here prevents Qt or Workbench from owning a second enum/cycle.
EXPECTED_PUBLIC_NAMES = (
    "PulseStreamer",
    "RemotePulseStreamer",
    "connect",
    "serve",
    "PulseSequence",
    "PulseApiParameter",
    "PulsePeriod",
    "AnalogStep",
    "PulsePortSpec",
    "PulseTarget",
    "PulseSlot",
    "PulseFieldRef",
    "OutputDelay",
    "MINIMUM_REPEAT_COUNT",
    "PULSE_TREE_FORMAT",
    "sequence_from_tree",
    "sequence_to_tree",
    "RepeatRegion",
    "compile_sequence",
    "pulse_target_from_xdc",
    "load_streamer_config",
    "UartRegisterTransport",
    "VivadoAxiRegisterTransport",
    "MemoryRegisterTransport",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "TIME_UNIT_CHOICES",
    "TIME_UNIT_TO_NS",
    "ANALOG_MODE_CHOICES",
    "align_to_grid",
    "cycle_binding_kind",
    "resolve_scan_point",
    "resolve_api_parameters",
    "pulse_field_value",
    "api_parameter_columns_for",
    "scan_columns_for",
    "scan_table_template",
    "validate_scan_table",
    "scan_rows_to_wire",
    "scan_rows_from_wire",
    "RemoteError",
    "UartError",
    "BackendResolutionError",
    "__version__",
)


def test_package_exports_are_the_final_surface_and_not_builtin_shadowing() -> None:
    names = tuple(zlc_pulse.__all__)
    assert names == EXPECTED_PUBLIC_NAMES
    # The exact listing is the contract; uniqueness catches accidental aliases.
    assert len(names) == len(set(names)) == len(EXPECTED_PUBLIC_NAMES)
    seen_ids: dict[int, str] = {}
    for name in names:
        value = getattr(zlc_pulse, name)
        if isinstance(value, int) and -5 <= value <= 256:
            continue
        previous = seen_ids.setdefault(id(value), name)
        assert previous == name, (previous, name)
        builtin_value = getattr(builtins, name, None)
        if builtin_value is not None:
            assert value is not builtin_value


def test_removed_implementation_types_stay_in_their_own_submodules() -> None:
    """Implementation, not surface -- and the list shrinks when that changes.

    MemoryRegisterTransport left it: it was listed as a test transport, and it
    stopped being one when the shipped window grew a "virtual" mode built on
    it.  The real command sequence with the wire substituted is a capability
    an operator uses, not a fixture, and a name returns to the facade the
    moment a sibling needs it for something the facade does not otherwise do.
    """

    for name in (
        "CompiledProgram",
        "RegisterTransport",
        "PulseRemoteServer",
        "StreamerParams",
        "AppliedState",
        "DoneReport",
        "SafeReadback",
        "trigger_times",
    ):
        assert not hasattr(zlc_pulse, name), name
    assert importlib.import_module("zlc_pulse.compile").CompiledProgram
    assert importlib.import_module("zlc_pulse.transport").RegisterTransport
    assert importlib.import_module("zlc_pulse.wire").StreamerParams
    assert importlib.import_module("zlc_pulse.device").AppliedState
    assert importlib.import_module("zlc_pulse.schedule").trigger_times
    assert not hasattr(wire, "PulseFieldRef")
    assert hasattr(remote, "PulseRemoteServer")


def test_contract_export_list_matches_package_export_list_in_both_directions() -> None:
    contract = (Path(__file__).resolve().parents[1] / "docs" / "contract.md").read_text(encoding="utf-8")
    block = contract.split("__all__ = (", 1)[1].split(")\n```", 1)[0]
    names = tuple(re.findall(r'"([A-Za-z_]\w*)"', block))
    assert names == EXPECTED_PUBLIC_NAMES
    assert set(names) == set(zlc_pulse.__all__)


def test_remote_adds_only_connection_management_methods() -> None:
    methods = {
        name
        for name, value in inspect.getmembers(RemotePulseStreamer, inspect.isfunction)
        if name in set(REMOTE_METHODS) | {"disconnect", "__enter__", "__exit__"}
    }
    assert methods - set(REMOTE_METHODS) == {"disconnect", "__enter__", "__exit__"}


def test_build_tools_live_in_the_fpga_submodule_not_package_exports() -> None:
    build = importlib.import_module("zlc_pulse.fpga")
    for name in ("emit_geometry_vh", "emit_geom_tcl", "estimate_resources", "solve_capacity", "check_config_capacity"):
        assert name in build.__all__
        assert name not in zlc_pulse.__all__
        assert not hasattr(zlc_pulse, name)


def test_host_compiled_program_does_not_expose_rtl_repeat_register() -> None:
    compiled = importlib.import_module("zlc_pulse.compile").CompiledProgram
    assert not hasattr(compiled, "repeat_from_index")
